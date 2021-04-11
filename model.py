#encoding:utf-8
from icecream import ic
import torch
import numpy as np
from torch.nn import Parameter, Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing, LayerNorm, GATConv
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax, to_dense_adj
from transformers import BertModel
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from transformers import BertConfig, RobertaConfig, XLNetConfig

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout, use_relu):
        super(MLP, self).__init__()
        self.use_relu = use_relu
        self.lin1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU(dropout)
        self.lin2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.lin1(x)
        if self.use_relu:
            x = self.relu(x)

        return self.lin2(x)


class Numerical_reason_layer(nn.Module):
    """
    model input
    in_size:768
    out_size:768
    """
    def __init__(self, in_size, out_size):
        super(Numerical_reason_layer, self).__init__()
        self.lin_1 = nn.Linear(in_size, out_size, bias= False)
        self.lin_2 = nn.Linear(256, out_size, bias= False)
    
    def forward(self, support, statement):
        re_support = self.lin_1(support)
        re_statement = self.lin_1(statement)
        statement_pooling = F.avg_pool1d(statement, kernel_size = 3)
        c = self.lin_2(statement_pooling)
        return re_support, re_statement, c

class GATNet(nn.Module):
    """
    in_size:768*4
    out_size:768
    """
    def __init__(self, node_dim):
        super(GATNet, self).__init__()
        self.node_dim = node_dim
        self.queries = nn.Linear(self.node_dim, self.node_dim)
        self.keys = nn.Linear(self.node_dim, self.node_dim)
        self.vals = nn.Linear(self.node_dim, self.node_dim)
        self.conv1 = GATConv(16*self.node_dim, self.node_dim, 4, dropout = 0.6)
        self.multiAttention = nn.MultiheadAttention(self.node_dim, num_heads=4)
    
    def expand_to_node_batch(self, node_batch, m):
        batch = node_batch.tolist()
        node_batch_index = set(batch)
        batch_len = []
        for bs in node_batch_index:
            batch_len.append(batch.count(bs))
        m_exp_list = []
        for i in range(len(batch_len)):
            m_exp_list.append(torch.unsqueeze(m[i], 0).repeat(batch_len[i], 1, 1))
        return torch.cat(m_exp_list, dim = 0), len(batch_len), batch_len
    
    def forward(self, x, edge_index, edge_attr, c, node_batch):
        m, batch_size, batch_list= self.expand_to_node_batch(node_batch, c)
        m = m.transpose(0, 1)
        x = x.reshape(m.size(1), -1, 768).transpose(0, 1)
        m_k = m 
        m_v = m
        att_x ,att_out_weight= self.multiAttention(x, m_k, m_v)
        att_x = att_x.transpose(0, 1).reshape(-1, 16*768)
        out = self.conv1(att_x, edge_index)
        return out, batch_size, batch_list

"""
class GAT(nn.Module):
    def __init__(self, node_dim):
        super().__init__()
        self.node_dim = node_dim
        self.stemDropout = 0.82
        self.readDropout = 0.85
        self.memoryDropout = 0.85
        self.alpha = 0.2
        self.iter_num = 4
        self.build_loc_x_init()
        self.build_propagate_message()

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def build_propagate_message(self):
        self.c_encoding = nn.Linear(self.node_dim, self.node_dim)
        self.trans_linear = nn.Linear(126, 16)
        self.readDropout = nn.Dropout(1 - self.readDropout)
        self.proj_x_loc = nn.Linear(self.node_dim, self.node_dim)
        self.proj_x_ctx = nn.Linear(self.node_dim, self.node_dim)
        self.queries = nn.Linear(3*self.node_dim, self.node_dim)
        self.keys = nn.Linear(3*self.node_dim, self.node_dim)
        self.vals = nn.Linear(3*self.node_dim, self.node_dim)
        self.proj_keys = nn.Linear(self.node_dim, self.node_dim)
        self.proj_vals = nn.Linear(self.node_dim, self.node_dim)
        self.mem_update = nn.Linear(2*self.node_dim, self.node_dim)
        self.combine_kb = nn.Linear(2*self.node_dim, self.node_dim)
    
    def expand_to_node_batch(self, node_batch, m):
        node_batch_index = set(node_batch)
        batch_len = []
        for bs in node_batch_index:
            batch_len.append(node_batch.count(bs))
        m_exp_list = []
        for i in range(len(batch_len)):
            #m_exp_list.append(torch.unsqueeze(m[i], 0).expand(batch_len[i], 3))
            m_exp_list.append(torch.unsqueeze(m[i], 0).repeat(batch_len[i], 1, 1))
        return torch.cat(m_exp_list, dim = 0), len(batch_len)

    def build_loc_x_init(self):
        self.init = nn.Linear(self.node_dim, self.node_dim)
        self.x_loc_drop = nn.Dropout(1-self.stemDropout)

        self.initMem = nn.Parameter(torch.randn(1, self.node_dim))
    
    def build_info_from_s_t(self, c):
        m = F.elu(self.c_encoding(c))
        return m
    
    def loc_ctx_init(self, x):
        x_loc = x
        x_ctx = self.initMem.expand(x_loc.size())

        x_ctx_drop = self.generate_scaler_drop_mask(x_ctx.size(), keep_prob=self.memoryDropout)
        '''
        mask = x_mask[:,:,None].expand(-1, -1, 1024).float()
        x_ctx_drop = x_ctx_drop*mask
        '''
        return x_loc, x_ctx, x_ctx_drop
    
    def generate_scaler_drop_mask(self, shape, keep_prob):
        assert keep_prob >0. and keep_prob <= 1
        mask = torch.rand(shape, device = 'cuda').le(keep_prob)
        mask = mask.float()/keep_prob
        return mask

    def forward(self, x, edge, edge_attr, node_batch, c):
        '''
        edge_attr:[batch_size, [node_edge]]
        '''
        ic(x.shape)
        node_batch = node_batch.tolist()
        x_loc, x_ctx, x_ctx_drop = self.loc_ctx_init(x)
        for t in range(self.iter_num):
            x_ctx = self.run_message_passing_iter(c, x_loc, x_ctx, x_ctx_drop, edge, node_batch, edge_attr)
        
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx]), dim = -1)

        return x_out
    
    def run_message_passing_iter(self, c, x_loc, x_ctx, x_ctx_drop, edge, node_batch, edge_attr):
        m = self.build_info_from_s_t(c)
        m = self.trans_linear(m.transpose(1, 2)).transpose(1, 2)
        x_ctx = self.propagate_message(m, x_loc, x_ctx, x_ctx_drop, edge, node_batch, edge_attr)
        return x_ctx

    def propagate_message(self, m, x_loc, x_ctx, x_ctx_drop, edge, node_batch, edge_attr):
        m, batch_size = self.expand_to_node_batch(node_batch, m)
        x_ctx = x_ctx * x_ctx_drop
        proj_x_loc = self.proj_x_loc(self.readDropout(x_loc))
        proj_x_ctx = self.proj_x_ctx(self.readDropout(x_ctx))
        x_joint = torch.cat([x_loc, x_ctx, proj_x_loc*proj_x_ctx], dim = -1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint)*self.proj_keys(m)[:,None,:]
        vals = self.vals(x_joint)*self.proj_vals(m)[:,None,:]
        N = keys.size(1)

        aij = torch.cat([queries.repeat(1, 1, N).view(batch_size, N*N, -1), keys.repeat(1, 1, batch_size, 1).view(batch_size, N*N, -1)], dim=-1).view(batch_size, N, N, -1)

        '''
        for i in range(self.edge_type_num):
          if (types >> i)&1==1:
            z = torch.matmul((aij*(edge.unsqueeze(-1)==i).float()), self.wes[i-1]).squeeze()
            edge_score += z
        '''
        e = self.leakyrelu(edge_score)#[bs, N, N]

        zero_vec = -9e15*torch.ones_like(e)#[N,N]
        edge_prob = torch.where(edge > 0, e, zero_vec)
        edge_prob = F.softmax(edge_prob, dim=-1)

        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        
        return x_ctx_new
"""


class Prediction_layer(nn.Module):
    """
    predict model
    """
    def __init__(self, in_size, mid_size, out_size):
        super(Prediction_layer, self).__init__()
        self.dropout = 0.5
        self.use_relu = True
        self.f = MLP(in_size, mid_size, out_size, self.dropout, self.use_relu)

    def forward(self, x, batch_list, batch_size):
        x = x.reshape(-1, batch_size, 768).transpose(0, 1)
        x_0 = self.f(x)
        x_1 = []
        batch_list.insert(0, 0)
        for i in range(batch_size):
            x_1.append(x_0[i][batch_list[i]:batch_list[i]+batch_list[i+1], :].max(dim=0)[0])
        x_1 = torch.stack(x_1)
        x_1_probs = F.softmax(x_1, dim = -1).max(dim = -1)[1]
        return x_1, x_1_probs
    
class TDGAN(nn.Module):
    def __init__(self, in_size, out_size):
        super(TDGAN, self).__init__()
        self.processor = nn.ModuleList([Numerical_reason_layer(in_size, out_size),
                                       #GAT(in_size),
                                       GATNet(in_size),
                                       Prediction_layer(768, 128, 2)])

    def forward(self, support, statement, graph_data):
        re_support, re_statement, c = self.processor[0](support, statement)
        graph_out, batch_size, batch_list = self.processor[1](graph_data.x, graph_data.edge_index, graph_data.edge_attr, c, next(iter(graph_data))[1])
        out, out_probs= self.processor[2](graph_out, batch_list, batch_size)
        return out, out_probs