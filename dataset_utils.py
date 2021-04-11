import json 
from tqdm import tqdm
from icecream import ic
from transformers import BertTokenizer, BertModel
import torch
import torch.nn
from utils import parse_opt
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device('cuda')

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
    将table中每一行的数据组成单个词的list
    对每个词都有一个对应的col索引
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
    将table中每一行的数据组成单个词的list
    对每个词都有一个对应的col索引
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
    确认：确保table中元素是statemen中的完整单词或词组
    查漏：statement中单个单词是否是table_element中的一部分单词
    """
    words = statement.split(" ")
    flag = 0
    for word in words:
        if word in table_element:
            flag = 1
    return flag


def delete_table_element_punc(table_element):
    """
    删除table元素的括号中的单词
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
    在table中寻找element_id
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
        [行， 列]
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
    在statement中寻找列名
    """
    col_element_ids = []
    #找到statemen中是否存在cols中的列名，返回[列， 列名]
    cols_element = match_col_statement(cols, statement)
    #在table——id中寻找列名下的元素id
    #table_element_id:[//第一个匹配的列名[row, col], //第二个匹配的列名[row, col], ...]
    table_element_id = []
    for element_id in cols_element:
        table_id = []
        table_id = numpy_where(element_id[0], table_sent_ids)
        table_element_id.append(table_id)
    #根据table_id在table中寻找元素
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
    将table_sen_id中的位置转为真实的table索引
    """
    table_id = []
    for element in element_id:
        table_id.append([element[0], table_sent_id[element[0]][element[1]]])
    return table_id

    

def search_entity(tag_data, table, cols):
    """
    挑选出statement中出现的实体和数字
    """
    num_type = ["NUMBER", "PERCENT", "MONEY", "TIME", "DATE", "DURATION", "ORDINAL"]
    #NER后识别出的实体
    num_entity_type = tag_data[2]
    #清洗num_type 以防出现 B-DATA这类标签
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
    #识别satement中的实体和数字，确定他们在table中的位置
    statement = tag_data[1]
    entities = []
    table_idxs = []
    entity_types = []
    table_element_ids = find_element_id_in_table(table_sents, statement)
    col_element_ids = find_element_id_in_cols(cols, table_sents, table_sent_ids, statement)
    #检测table_element_ids和col_element_ids是否有重复的table——element
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
