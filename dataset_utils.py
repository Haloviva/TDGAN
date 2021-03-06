import json 
from tqdm import tqdm
from icecream import ic
from transformers import BertTokenizer, BertModel
import torch
import torch.nn
from utils import parse_opt
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device('cuda')

def Encoding(dataset, data_type):
    print(20*"=" + "pre encoding"+ 10*'=')
    args = parse_opt()
    num_data = dataset.__len__()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    #model = torch.nn.DataParallel(model)
    model = DDP(model)
    supports = []
    statements = []
    tables = []
    columns = []
    labels = []
    for i in tqdm(range(num_data)):
        data_dict = dataset.__getitem__(i)
        supports.append(data_dict['support'])
        statements.append(data_dict['statement'])
        labels.append(data_dict['label'])
        columns.append(data_dict['column']) 
        table = data_dict['table']
        table_cell_list = []
        for row in range(len(table)):
            for col in range(len(table[row])):
                table_cell_list.append(table[row][col])
        tables.append(table_cell_list)
    split_data_num = num_data//(args.batch)
    split_list = []
    for n in range(args.batch-1):
        split_list.append(split_data_num*n)
    split_list.append(num_data)
    for j in tqdm(range(len(split_list)-1)):
        support_tokens = tokenizer(supports[split_list[j]:split_list[j+1]], padding = 'max_length', truncation = True, max_length = args.sent_max_len, return_tensors='pt')
        statement_tokens = tokenizer(statements[split_list[j]:split_list[j+1]], padding = 'max_length', truncation = True, max_length = args.sent_max_len, return_tensors='pt')
        '''
        support_tokens = support_tokens.cuda()
        statement_tokens = statement_tokens.cuda()
        '''
        support_tokens.to(device)
        statement_tokens.to(device)
        support_out = model(**support_tokens).last_hidden_state
        statement_out = model(**statement_tokens).last_hidden_state
        column_out = []
        #print(20*"="+"encoding column"+20*"=")
        for column in tqdm(columns[split_list[j]:split_list[j+1]]):
            column_tokens = tokenizer(column, padding = 'max_length', truncation = True, max_length = args.table_max_len, return_tensors='pt')
            #column_tokens = column_tokens.cuda()
            column_tokens.to(device)
            out = model(**column_tokens).last_hidden_state
            column_out.append(out)
        table_out = []
        #print(20*"="+"encoding table"+20*"=")
        for table in tqdm(tables[split_list[j]:split_list[j+1]]):
            table_tokens = tokenizer(table, padding = 'max_length', truncation = True, max_length = args.table_max_len, return_tensors='pt')
            #table_tokens = table_tokens.cuda()
            table_tokens.to(device)
            out = model(**table_tokens).last_hidden_state
            table_out.append(out)
        data = []
        min_data = []
        mini_num = len(support_out)
        for m in range(mini_num):
            min_data.append(support_out[m])
            min_data.append(statement_out[m])
            min_data.append(table_out[m])
            min_data.append(column_out[m])
            min_data.append(torch.tensor(labels[m]))
            data.append(min_data)
        torch.save(data, args.data+"/processed_datasets/embedding_data/{}_part0_{}.pt".format(data_type, j))

def Encoding_sentence(sent_data, data_type, id):
    args = parse_opt()
    num_sent = len(sent_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model = DDP(model)
    supports = []
    statements = []
    for i in range(num_sent):
        supports.append(sent_data[i][1])
        statements.append(sent_data[i][2])
    support_tokens = tokenizer(supports, padding = 'max_length', truncation = True, max_length = args.sent_max_len, return_tensors = 'pt')
    statement_tokens = tokenizer(statements, padding = 'max_length', truncation = True, max_length = args.sent_max_len, return_tensors='pt')
    support_tokens.to(device)
    statement_tokens.to(device)
    support_out = model(**support_tokens).last_hidden_state
    statement_out = model(**statement_tokens).last_hidden_state
    torch.save(support_out, args.data+"/processed_datasets/embedding_data/{}_support_{}.pt".format(data_type, id))
    torch.save(statement_out, args.data+"/processed_datasets/embedding_data/{}_statement_{}.pt".format(data_type, id))

def Encoding_table(table_data, data_type, id):
    args = parse_opt()
    num_table = len(table_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model = DDP(model) 
    tables = []
    for i in range(num_table):
        table = table_data[i]
        table_cell_list = []
        for row in range(len(table)):
            for col in range(len(table[row])):
                table_cell_list.append(table[row][col])
        tables.append(table_cell_list)
    table_out = []
    for m in range(num_table):
        table_tokens = tokenizer(tables[m], padding = 'max_length', truncation = True, max_length = args.table_max_len, return_tensors='pt')
        table_tokens.to(device)
        out = model(**table_tokens).last_hidden_state
        table_out.append(out)
    torch.save(table_out, args.data+"/processed_datasets/embedding_data/{}_table_{}.pt".format(data_type, id))

def Encoding_column(column_data, data_type, id):
    args = parse_opt()
    num_column = len(column_data)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model = DDP(model) 
    columns = []
    for i in range(num_column):
        columns.append(column_data[i])
    column_out = []
    for m in range(num_column):
        column_tokens = tokenizer(columns[m], padding = 'max_length', truncation = True, max_length = args.table_max_len, return_tensors='pt')
        column_tokens.to(device)
        out = model(**column_tokens).last_hidden_state
        column_out.append(out)
    torch.save(column_out, args.data+"/processed_datasets/embedding_data/{}_column_{}.pt".format(data_type, id))

def read_pt(data_path, data_type, table_data, labels):
    data = []
    num_data = len(table_data)
    supports = []
    statements = []
    columns = []
    for i in tqdm(range(64)):
        support = torch.load(data_path+"/processed_datasets/embedding_data/{}_support_{}.pt".format(data_type, i))
        statement = torch.load(data_path+"/processed_datasets/embedding_data/{}_statement_{}.pt".format(data_type, i))
        column = torch.load(data_path+"/processed_datasets/embedding_data/{}_column_{}.pt".format(data_type, i))
        for j in range(len(support)):
            supports.append(support[j])
            statements.append(statement[j])
            columns.append(column[j])
    for j in range(len(supports)):
        mini_data = []
        mini_data.append(supports[j])
        mini_data.append(statements[j])
        mini_data.append(table_data[j])
        mini_data.append(columns[j])
        mini_data.append(int(labels[j]))
        data.append(mini_data)
    return data

def convert_table_to_sent(table):
    """
    ???table???????????????????????????????????????list
    ?????????????????????????????????col??????
    """
    table_sents = []
    table_sent_ids = []
    for row in range(len(table)):
        sent = []
        sent_id = []
        for col in range(len(table[row])):
            element = table[row][col].split(' , ')
            sent.extend(element)
            sent_id.extend([col]*len(element))
        table_sents.append(sent)
        table_sent_ids.append(sent_id)
    return table_sents,table_sent_ids



def convert_table_to_sent(table):
    """
    ???table???????????????????????????????????????list
    ?????????????????????????????????col??????
    """
    table_sents = []
    table_sent_ids = []
    for row in range(len(table)):
        sent = []
        sent_id = []
        for col in range(len(table[row])):
            element = table[row][col].split(' , ')
            sent.extend(element)
            sent_id.extend([col]*len(element))
        table_sents.append(sent)
        table_sent_ids.append(sent_id)
    return table_sents,table_sent_ids

def comfirm_element(table_element, statement):
    """
    ???????????????table????????????statemen???????????????????????????
    ?????????statement????????????????????????table_element?????????????????????
    """
    words = statement.split(" ")
    flag = 0
    for word in words:
        if word in table_element:
            flag = 1
    return flag


def delete_table_element_punc(table_element):
    """
    ??????table???????????????????????????
    """

    if "(" in table_element:
        table_words = table_element.split(' (')
        table_ele = table_words[0]
    elif "-" in table_element:
        table_words = table_element.split(' -')
        table_ele = table_words[0]
    else:
        table_ele = table_element
    return table_ele


def find_element_id_in_table(table_sents, statement):
    '''
    ???table?????????element_id
    element_ids
    [
        [[row, col], table_element],
        []
    ]
    '''
    element_ids = []
    for row in range(len(table_sents)):
        indx = []
        for col in range(len(table_sents[row])):
            table_ele = delete_table_element_punc(table_sents[row][col])
            #if comfirm_element_in_statement(table_sents[row][col], statement) and comfirm_element(table_sents[row][col], statement):
            if statement.find(table_ele) != -1 and comfirm_element(table_ele, statement):
                element = [[row, col], table_sents[row][col]]
                indx.append(element)
        if len(indx) != 0:
            element_ids.append(indx)
    return element_ids

def numpy_where(idx, id_list):
    """
    finded:
    [
        [?????? ???]
    ]
    """
    finded = []
    for i in range(len(id_list)):
        if idx in id_list[i]:
            for j in range(len(id_list[i])):
                if id_list[i][j] == idx:
                    finded.append([i, j])
    return finded

def match_col_statement(cols, statement):
    words = statement.split(' ')
    indx = []
    for col in cols:
        col_words = col.split(' ')
        col_id = cols.index(col)
        try:
            for word in col_words:
                if word in words:
                    idx = [col_id, col]
                    indx.append(idx)
                    idx = []
        except UnboundLocalError:
            continue
    return indx
        

def find_element_id_in_cols(cols, table_sents, table_sent_ids, statement):
    """
    ???statement???????????????
    """
    col_element_ids = []
    #??????statemen???????????????cols?????????????????????[?????? ??????]
    cols_element = match_col_statement(cols, statement)
    #???table??????id???????????????????????????id
    #table_element_id:[//????????????????????????[row, col], //????????????????????????[row, col], ...]
    table_element_id = []
    for element_id in cols_element:
        table_id = []
        table_id = numpy_where(element_id[0], table_sent_ids)
        table_element_id.append(table_id)
    #??????table_id???table???????????????
    for col in table_element_id:
        for element_id in col:
            indx = []
            row = element_id[0]
            col = element_id[1]
            idx = [row, table_sent_ids[row][col]]
            indx.append(idx)
            indx.append(table_sents[row][col])
            col_element_ids.append(indx)

    return col_element_ids


def convert_element_id_to_table_id(table_sent_id, element_id):
    """
    ???table_sen_id???????????????????????????table??????
    """
    table_id = []
    for element in element_id:
        table_id.append([element[0], table_sent_id[element[0]][element[1]]])
    return table_id

    

def search_entity(tag_data, table, cols):
    """
    ?????????statement???????????????????????????
    """
    num_type = ["NUMBER", "PERCENT", "MONEY", "TIME", "DATE", "DURATION", "ORDINAL"]
    #NER?????????????????????
    num_entity_type = tag_data[2]
    #??????num_type ???????????? B-DATA????????????
    for entity in num_entity_type:
        for n_type in num_type:
            if n_type in entity[1]:
                entity[1] = n_type
    num_entity = []
    num_types = []
    for entity in num_entity_type:
        if entity[0] != '[UNK]' and entity[1] in num_type:
            num_entity.append(entity[0])
            num_types.append(entity[1])
    
    table_sents, table_sent_ids = convert_table_to_sent(table)
    #??????satement???????????????????????????????????????table????????????
    statement = tag_data[1]
    entities = []
    table_idxs = []
    entity_types = []
    table_element_ids = find_element_id_in_table(table_sents, statement)
    col_element_ids = find_element_id_in_cols(cols, table_sents, table_sent_ids, statement)
    #??????table_element_ids???col_element_ids??????????????????table??????element
    if len(table_element_ids) or len(col_element_ids):
        element_ids = []
        for table_element in table_element_ids:
            element_ids.append(table_element[0])
        for col_element in col_element_ids:
            if col_element not in element_ids:
                element_ids.append(col_element)
        for element in element_ids:
            word = element.pop()
            entities.append(word)
            table_id = convert_element_id_to_table_id(table_sent_ids, element)
            table_idxs.append(table_id)
            if word not in num_entity:
                entity_types.append("ENTITY")
            else:
                #ic(word)
                #ic(num_entity)
                #ic(num_type)
                idx = num_entity.index(word)
                entity_types.append(num_types[idx])
                num_entity.pop(idx)
                num_types.pop(idx)
    elif len(num_entity) != 0:
        for i in range(len(num_entity)):
            entities.append(num_entity[i])
            entity_types.append(num_types[i])

    return entities, table_idxs, entity_types
