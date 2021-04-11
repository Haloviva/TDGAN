#encoing:utf-8
import json
import pandas
import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
from dataset_utils import *


class LoadData():
    def __init__(self, filename, data_type):
        self.filename = filename
        self.data_type = data_type

    def read_tsv(self):
        """
        读取所有data
        data结构：
        data{
            [id, support, statement,label]---instance
        }
        """

        self.data = []
        self.table_ids = []

        with open(self.filename+'/processed_datasets/{}.tsv'.format(self.data_type),"r") as f:
            lines = f.readlines()
            instances = []
            print("Reading {} data ...".format(self.data_type))
            for line in tqdm(lines):
                instance = []
                content = line.strip().split("\t")
                table_id = content[0]
                self.table_ids.append(table_id)
                support = content[3]
                statement = content[4]
                label = content[5]
                instance.append(table_id)
                instance.append(support)
                instance.append(statement)
                instance.append(label)
                instances.append(instance)
                self.data.append(instance)
        return instances, self.table_ids
                
    def read_table(self, table_ids):
        """
        table_data
            "table":[
                //table_1
                [
                    [table第一列元素],
                    [table第二列元素],
                    ...
                ]
                //table_2
                [
                    [table第一列元素],
                    [table第二列元素],
                    ...
                ]
            ]
            "cols":[
                //table_1
                [列名1，列名2，...],
                //table_2
                [列名1，列名2，...],
                ...
            ]   
        }
        """
        self.table_data = {}
        tab = []
        cols = []
        print("Reading table data ...")
        for id in tqdm(table_ids):
            table = pandas.read_csv(self.filename+'/all_csv/{}'.format(id),'#')
            _cols = []
            for col in table.columns:
                _cols.append(col)
            rows = []
            for i in range(len(table)):
                row = []
                for j in range(len(_cols)):
                    row.append(str(table.iloc[i][j]))
                rows.append(row)
            tab.append(rows)
            cols.append(_cols)

        self.table_data['table'] = tab
        self.table_data['column'] = cols
        return self.table_data


class TabFactDataset_with_no_edge_weight(Dataset):
    def __init__(self, 
                sent_data,
                graph_data):

        self.num_data = len(sent_data)

        self.data = {"id":[],
                    "support": [],
                    "statement": [],
                    "label": []
        }
        
        for i in tqdm(range(len(sent_data))):
            instance = sent_data[i]
            self.data["id"].append(instance[0])
            self.data["support"].append(instance[1])
            self.data["statement"].append(instance[2])
            self.data["label"].append(int(instance[3]))

        self.data["table"] = graph_data['table']
        self.data["column"] = graph_data['column']

    def __len__(self):
        return self.num_data
    
    def __getitem__(self,index):
        return {"id": self.data["id"][index],
                "support": self.data["support"][index],
                "statement": self.data["statement"][index],
                "label": self.data["label"][index],
                "table": self.data["table"][index],
                "column": self.data["column"][index]
        }


class EmbeddingDataset(Dataset):
    '''
    mini_data:
    [id, supprot, statement, label, table, column]
    self.data:
    [mini_data,...]
    '''
    def __init__(self, datapath, datatype, table_data):
        with open(datapath+"/processed_datasets/embedding_data/{}_label.json".format(datatype),'r') as f:
            labels = json.load(f)
        self.data = read_pt(datapath, datatype, table_data, labels)
        self.num_data = len(self.data)
    
    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        return self.data[index]

    