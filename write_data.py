from torch.utils.data import DataLoader as torch_DataLoader
from dataset import *
from dataset_utils import Encoding
from run import read_dataset
from utils import parse_opt

def write_dataset(data_type):
    datapath = '/home/DATA/TabFact'

    dataloader = LoadData(datapath, data_type)
    sentdata, train_table_ids = dataloader.read_tsv()
    table_data =  dataloader.read_table(table_ids)
    
    with open(datapath+'/processed_datasets/{}_sentdata.json'.format(data_type), 'w') as f:
        json.dump(sent_data,f)
    f.close()

    with open(datapath+'/processed_datasets/{}_table_ids.json'.format(data_type), 'w') as f:
        json.dump(table_ids,f)
    f.close()

    with open(datapath+'/processed_datasets/{}_table_data.json'.format(data_type), 'w') as f:
        json.dump(table_data, f)
    f.close()

def write_embedding_sent_dataset(data_type, id):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    sent_data_part = []
    for i in range(176*id, 176*(id+1)):
        sent_data_part.append(sent_data[i])
    '''
    tabfact_dataset = TabFactDataset_with_no_edge_weight(sent_data_part, table_data_part)
    TabFactDataLoader = (tabfact_dataset)
    data = Encoding(TabFactDataLoader, data_type)
    #torch.save(data, datapath+"/processed_datasets/embedding_data/{}.pt".format(data_type))
    '''
    Encoding_sentence(sent_data_part,'train', id)

def write_embedding_sent_dataset_end(data_type):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    sent_data_part = []
    for i in range(90112, 90233):
        sent_data_part.append(sent_data[i])
    Encoding_sentence(sent_data_part, 'train', '512')

def write_embedding_table_dataset(data_type, id):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    table_data_part = []
    for i in range(176*id, 176*(id+1)):
        table_data_part.append(table_data["table"][i])
    '''
    tabfact_dataset = TabFactDataset_with_no_edge_weight(sent_data_part, table_data_part)
    TabFactDataLoader = (tabfact_dataset)
    data = Encoding(TabFactDataLoader, data_type)
    #torch.save(data, datapath+"/processed_datasets/embedding_data/{}.pt".format(data_type))
    '''
    Encoding_table(table_data_part,'train', id)

def write_embedding_table_dataset_end(data_type):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    table_data_part = []
    for i in range(90112, 90233):
        table_data_part.append(table_data["table"][i])
    Encoding_table(table_data_part, 'train', '512')

def write_embedding_column_dataset(data_type, id):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    column_data_part = []
    for i in range(176*id, 176*(id+1)):
        column_data_part.append(table_data["column"][i])
    '''
    tabfact_dataset = TabFactDataset_with_no_edge_weight(sent_data_part, table_data_part)
    TabFactDataLoader = (tabfact_dataset)
    data = Encoding(TabFactDataLoader, data_type)
    #torch.save(data, datapath+"/processed_datasets/embedding_data/{}.pt".format(data_type))
    '''
    Encoding_column(column_data_part,'train', id)

def write_embedding_column_dataset_end(data_type):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    column_data_part = []
    for i in range(90112, 90233):
        column_data_part.append(table_data["column"][i])
    Encoding_column(column_data_part, 'train', '512')

def write_labels(data_type):
    datapath = '/home/DATA/TabFact'
    sent_data, table_ids, table_data = read_dataset(datapath, data_type)
    num_data = len(sent_data)
    labels = []
    for i in range(num_data):
        labels.append(sent_data[i][-1])
    with open(datapath+'/processed_datasets/embedding_data/{}_label.json'.format(data_type), 'w') as f:
        json.dump(labels, f)
    f.close()


if __name__ == "__main__":
    args = parse_opt()
    '''
    write_dataset("train")
    write_dataset("test")
    write_dataset("dev")
    write_dataset("example")
    '''
    #write_embedding_dataset("example")
    '''
    for i in tqdm(range(512)):
        write_embedding_sent_dataset('train', i)
    write_embedding_sent_dataset_end('train')
    '''
    '''
    for j in tqdm(range(512)):
        write_embedding_table_dataset('train', j)
    write_embedding_table_dataset_end('train')
    '''
    '''
    for j in tqdm(range(512)):
        write_embedding_column_dataset('train', j)
    '''
    #write_embedding_column_dataset_end('train')
    write_labels("train")
    