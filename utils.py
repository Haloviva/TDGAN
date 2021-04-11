#encoding:utf-8
import tqdm
import argparse
import json
import time
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data, DataLoader, Batch
from tqdm import tqdm
from icecream import ic
from model import *


device = torch.device('cuda')
dist.init_process_group(backend = 'nccl', init_method="env://")

def parse_opt():
    """
    接受参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/DATA/TabFact',type=str)
    parser.add_argument('--pretrained_model', default='bert-base-uncased', type=str)
    parser.add_argument('--dim', default=768, type=int)
    parser.add_argument('--head', default=4, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch', default=4, type=int)
    parser.add_argument('--split', default=256, type=int)
    parser.add_argument('--sent_max_len', default=126, type=int)
    parser.add_argument('--table_max_len', default=16, type=int)
    parser.add_argument('--max_grad_norm', default=5, type=int)
    parser.add_argument('--do_example', default=False, action='store_true')
    parser.add_argument('--do_train', default=False, action="store_true")
    parser.add_argument('--do_test', default=False, action="store_true")
    parser.add_argument('--do_dev', default=False, action="store_true")
    parser.add_argument('--edge', default=False, action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int, help="Local rank for distributed training.")
    parser.add_argument('--lr_default', type=float, default=5e-5)
    parser.add_argument('--load_from', default='', type=str)
    parser.add_argument('--target_dir', default='./model_save', type=str)

    args = parser.parse_args()
    
    return args

def correct_predictions(probs, labels):
    correct = 0
    for i in range(len(labels)):
        if probs[i] == labels[i]:
            correct += 1
    correct = correct/len(labels)
    return correct

def train(model, 
        dataloader,
        optimizer,
        criterion):
    """
    """
    args = parse_opt()
    model.train()

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0


    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        logits = [] 
        probs = []
        labels = []
        supports = []
        statements = []
        columns = []
        if args.edge:
            entities = []
            entity_table_indexs = []
            entity_types = []
            for bat in batch:
                labels.append(bat["label"])
                supports.append(bat["support"])
                statements.append(bat["statement"])
                entities.append(bat["entity"])
                entity_table_indexs.append(bat["entity_table_indexs"])
                entity_types.append(bat["entity_type"])
                columns.append(bat["column"])
            labels = torch.tensor(labels).to(device)
            '''
            support_output, statement_output, support_mask, statement_mask = sentencoded(supports, statements)
            support_output.to(device)
            statement_output.to(device)
            '''
            #带边图
            table_datas = form_graph_with_edge_weight(entities, entity_table_indexs, entity_types, columns)
            graph_data = table_datas.to(device)
        else:
            tables = []
            for bat in batch:
                labels.append(bat[-1])
                supports.append(bat[0].unsqueeze(dim=0))
                statements.append(bat[1].unsqueeze(dim=0))
                tables.append(bat[2])
                columns.append(bat[3])
            labels = torch.tensor(labels).to(device)
            support_out = torch.cat(supports, dim = 0).to(device)
            statement_out = torch.cat(statements, dim = 0).to(device)
            
            #support_out, statement_out, support_mask, statement_mask = sentencoded(supports, statements)
            support_out.to(device)
            statement_out.to(device)
            

            table_datas = form_graph_with_no_edge_weight(tables, columns)
            graph_data = table_datas.to(device) 

        optimizer.zero_grad()
        logits, probs = model(support_out,
                            statement_out,
                            graph_data)
    
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)

        description = "Arg. batch proc. time: {:.4f}s, loss: {:.4f}".format(batch_time_avg/(batch_index+1),
                        running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() 
    epoch_start = epoch_time
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    
    return epoch_time, epoch_loss, epoch_accuracy
    

def validate(model, dataloader, criterion):
    """
    """
    model.eval()

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            supports = batch[batch_index]["support"]
            #supports_length = batch["support_length"].to(device)
            statements = batch[batch_index]["statement"]
            #statements_length = batch["statements_length"].to(device)
            labels = batch[batch_index]["label"]
            
            support_output, statement_output = sentencoded(supports, statements)
            support_output.to(device)
            statement_output.to(device)
            labels.to(device)

            table_data = form_graph(batch[batch_index]["entity"], batch[batch_index]["entity_table_indexs"], batch[batch_index]["entity_type"], batch[batch_index]["column"])

            graph_data = table_data["x"].to(device)
            graph_edge = table_data["edge_index"].to(device)

            logits, probs = model(support_output,
                                  statement_output,
                                  graph_data,
                                  graph_edge)
            loss = criterion(logits, labels)

            _, out_label = probs.max(dim=1)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)
    
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy

def test(model, dataloader):
    """
    """
    model.eval()

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0

    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for batch_index, batch in enumerate(tqdm_batch_iterator):
            supports = batch[batch_index]["support"]
            statements = batch[batch_index]["statements"]
            labels = batch[batch_index]["labels"]

            support_output, statement_output = sentencoded(supports, statements)
            support_output.to(device)
            statement_output.to(device)
            labels.to(device)

            table_data = form_graph(batch[batch_index]["entity"], batch[batch_index]["entity_table_indexs"], batch[batch_index]["entity_type"], batch[batch_index]["column"]) 
            graph_data = table_data["x"].to(device)
            graph_edge = table_data["edge_index"].to(device)
            
            _, probs = model(support_output,
                             statement_output,
                             graph_data,
                             graph_edge)
            
            accuracy += correct_predictions(probs, labels)
            batch_time += time.time() - batch_start
        
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy

'''
def sentencoded(support, statement):
    """
    对support，statement编码
    """
    print("===============doing sentence encode ...==============")
    start_time = time.time()
    args = parse_opt()
    model_type = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_type)
    supports_token = tokenizer(support, padding = 'max_length', truncation = True, max_length = args.max_len, return_tensors = "pt")
    statements_token = tokenizer(statement, padding = 'max_length', truncation = True, max_length = args.max_len, return_tensors = "pt")

    supports_mask = supports_token.attention_mask
    statements_mask = statements_token.attention_mask

    supports_token.to(device)
    statements_token.to(device)

    model = BertModel.from_pretrained(model_type)
    model.to(device)
    supports_outputs = model(**supports_token)    
    statements_outputs = model(**statements_token)
    end_time = time.time()
    continue_time = end_time - start_time
    ic(continue_time)

    return supports_outputs.last_hidden_state, statements_outputs.last_hidden_state, supports_mask, statements_mask
'''

def form_graph_with_edge_weight(entities, entity_table_indexs, entity_type, cols):
    """
    使用替换数字后的table 以及statement 构图
    str实体 数字类型 作为顶点
    col 作为实体与数字类型的
    数字类型 作为数字类型相同的顶点的边
    table:
    [   
        //table_1
        [
            [第一行元素]
            [...]
            ...
        ]
        //table_2
        [

        ]
    ]
    cols: 
    [
        //table_1
        [列名1，列名2，...]
        //table_2
        []
        ...
    ]
    tag:
    [
        [id, statement,[number, number_type]]
    ]
    """
    args = parse_opt()
    model_type = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_type)
    data = []
        
    batch_len = len(entities)
    
    entity_digit_tokens = tokenizer("ENT+DIGIT", padding = 'max_length', truncation = True, max_length = 16, return_tensors = "pt")

    entity_digit_tokens.to(device)

    model = BertModel.from_pretrained(model_type)
    model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters = True)

    entity_digit_output = model(**entity_digit_tokens).last_hidden_state
    
    entity_digit_output.to(device)

    for idx in range(batch_len):
        entity_len = len(entities[idx])

        np_entity_type = np.array(entity_type[idx])
        entity_number = np.where(np_entity_type == "ENTITY", 1, 0)


        entity_tokens = tokenizer(entities[idx], padding = 'max_length', truncation = True, max_length = 16, return_tensors = "pt")
        entity_type_tokens = tokenizer(entity_type[idx], padding = 'max_length', truncation = True, max_length = 16, return_tensors = "pt")
        cols_tokens = tokenizer(cols[idx], padding = 'max_length', truncation = True, max_length = 16, return_tensors = "pt")

        entity_tokens.to(device)
        entity_type_tokens.to(device)
        cols_tokens.to(device)

        entity_output = model(**entity_tokens).last_hidden_state
        entity_type_output = model(**entity_type_tokens).last_hidden_state
        cols_output = model(**cols_tokens).last_hidden_state
        
        entity_output.to(device)
        entity_type_output.to(device)

        entity_output = entity_output.view(entity_len, -1)


        #edge_weight = torch.zeros([entity_len, entity_len, 16, args.dim], dtype = torch.long)
        edge_idx = []
        edge_weight = []

        for i in range(entity_len):
            for j in range(1, entity_len):
                edge = []
                if not (entity_number[i] and entity_number[j]):
                    edge_weight_entity_idx = i if entity_number[i] == 0 else j
                    try: 
                        edge_weight_entity_col = cols_output[entity_table_indexs[idx][edge_weight_entity_idx][1]+1]
                        edge = [i, j]
                        edge_idx.append(edge)
                        edge = [j, i]
                        edge_idx.append(edge)
                        #ic(entity_weight_entity_col.shape)
                        edge_weight.append(entity_weight_entity_col)
                        edge_weight.append(entity_weight_entity_col)
                    except IndexError:
                        edge = [i, j]
                        edge_idx.append(edge)
                        edge = [j, i]
                        edge_idx.append(edge)
                        #ic(entity_digit_output.shape)
                        edge_weight.append(entity_digit_output)
                        edge_weight.append(entity_digit_output)
                elif (entity_number[i] == 1 and (entity_type[idx][i] == entity_type[idx][j])):
                    edge = [i, j]
                    edge_idx.append(edge)
                    edge = [j, i]
                    edge_idx.append(edge)
                    #ic(entity_type_output.shape)
                    edge_weight.append(entity_type_output[i])
                    edge_weight.append(entity_type_output[i])
        edge_index = torch.tensor(edge_idx, dtype = torch.long)
        edge_weight_len = len(edge_weight)
        if edge_weight_len != 0:
            edge_weights = torch.cat(edge_weight, dim = 0).view(edge_weight_len,-1)
            data.append(Data(x = entity_output, edge_index = edge_index.t(), edge_attr = edge_weights))
        else:
            data.append(Data(x = entity_output, edge_index = edge_index.t()))
    dataset = Batch.from_data_list(data)
    #dataset = DataLoader(data, batch_size = len(data))
    return dataset

'''
def form_graph_with_no_edge_weight(tables, cols):
    data = []
    batch_size = len(tables)

    for idx in range(batch_size):
        col = len(cols[idx])
        row = int(tables[idx].size(0)/col)
        table = tables[idx].reshape(row*col, -1)

        edge_index = []
        for m in range(row):
            for i in range(col):
                for j in range(i+1, col):
                    edge = [col*m+i, col*m+j]
                    edge_index.append(edge)
                    edge = [col*m+j, col*m+i]
                    edge_index.append(edge)
        for m_ in range(col):
            for i_ in range(m_, m_+row*col, col):
                for j_ in range(i_+col, m_+row*col, col):
                    edge = [i_, j_]
                    edge_index.append(edge)
                    edge = [j_, i_]
                    edge_index.append(edge)
        edge_index = torch.tensor(edge_index,dtype= torch.long)
        data.append(Data(x = table, edge_index=edge_index.t()))
    ic(data)
    dataset = Batch.from_data_list(data)
    return dataset
'''

def form_graph_with_no_edge_weight(tables, cols):
    args = parse_opt()
    model_type = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_type)
    model = BertModel.from_pretrained(model_type)
    model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters = True)

    data = []
    batch_size = len(tables)

    for idx in range(batch_size):
        row = len(tables[idx])
        col = len(cols[idx])

        table_cell = []
        for row_cell in tables[idx]:
            for cell in row_cell:
                table_cell.append(cell)

        table_cell_tokens = tokenizer(table_cell, padding = 'max_length', truncation = True, max_length = 16, return_tensors = "pt")
        
        table_cell_tokens.to(device)

        table_cell_outputs = model(**table_cell_tokens).last_hidden_state
        table_cell_outputs = table_cell_outputs.reshape(row*col, -1)

        edge_index = []
        for m in range(row):
            for i in range(col):
                for j in range(i+1, col):
                    edge = [col*m+i, col*m+j]
                    edge_index.append(edge)
                    edge = [col*m+j, col*m+i]
                    edge_index.append(edge)
        for m_ in range(col):
            for i_ in range(m_, m_+row*col, col):
                for j_ in range(i_+col, m_+row*col, col):
                    edge = [i_, j_]
                    edge_index.append(edge)
                    edge = [j_, i_]
                    edge_index.append(edge)
        edge_index = torch.tensor(edge_index,dtype= torch.long)
        data.append(Data(x = table_cell_outputs, edge_index=edge_index.t()))
    dataset = Batch.from_data_list(data)
    return dataset


        



        
    