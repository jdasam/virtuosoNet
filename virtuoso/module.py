import torch
import torch.nn as nn
from . import model_utils as utils
from .model_utils import combine_splitted_graph_output

class GatedGraph(nn.Module):
    def  __init__(self, size, num_edge_style, secondary_size=0):
        super(GatedGraph, self).__init__()
        if secondary_size == 0:
            secondary_size = size
        self.size = size
        self.secondary_size = secondary_size

        self.ba = torch.nn.Parameter(torch.Tensor(size))
        self.wz_wr_wh = torch.nn.Parameter(torch.Tensor(num_edge_style,size,secondary_size * 3))
        self.uz_ur = torch.nn.Parameter(torch.Tensor(size, secondary_size * 2))
        self.uh = torch.nn.Parameter(torch.Tensor(secondary_size, secondary_size))

        nn.init.xavier_normal_(self.wz_wr_wh)
        nn.init.xavier_normal_(self.uz_ur)
        nn.init.xavier_normal_(self.uh)
        nn.init.zeros_(self.ba)

    def forward(self, input, edge_matrix, iteration=10):
        for i in range(iteration):
            if edge_matrix.shape[0] != self.wz_wr_wh.shape[0]:
                # splitted edge matrix
                num_batch = edge_matrix.shape[0]//self.wz_wr_wh.shape[0]
                num_type = self.wz_wr_wh.shape[0]
                input_split = utils.split_note_input_to_graph_batch(input, num_batch, edge_matrix.shape[1])
                # num_notes = edge_matrix.shape[1]
                # split_num = num_notes * 2 //3
                # edge_matrix_split = torch.cat([edge_matrix[:,:split_num,:split_num],
                #                             edge_matrix[:,-split_num:, -split_num: ]], dim=0)
                # input_split = torch.cat([input[:,:split_num,:],
                #                             input[:,-split_num:,: ]], dim=0)
                activation_split = torch.bmm(edge_matrix.transpose(1,2), input_split.repeat(1,self.wz_wr_wh.shape[0],1).view(-1,edge_matrix.shape[1],input.shape[2])) + self.ba
                activation_wzrh_split = torch.bmm(activation_split, self.wz_wr_wh.repeat(input_split.shape[0],1,1))
                activation_wz_split, activation_wr_split, activation_wh_split = torch.split(activation_wzrh_split, self.secondary_size, dim=-1)
                input_uzr_sp = torch.matmul(input_split, self.uz_ur)
                input_uz_sp, input_ur_sp = torch.split(input_uzr_sp, self.secondary_size, dim=-1)
                temp_z_sp = torch.sigmoid(activation_wz_split.view(num_batch,num_type,input_split.shape[1],-1).sum(1)+input_uz_sp)
                temp_r_sp = torch.sigmoid(activation_wr_split.view(num_batch,num_type,input_split.shape[1],-1).sum(1)+input_ur_sp)

                if self.secondary_size == self.size:
                    temp_hidden_sp = torch.tanh(
                        activation_wh_split.view(num_batch,num_type,input_split.shape[1],-1).sum(1) + torch.matmul(temp_r_sp * input_split, self.uh))
                    input_split = (1 - temp_z_sp) * input_split + temp_r_sp * temp_hidden_sp
                    input = combine_splitted_graph_output(input_split, input)
                else:
                    temp_hidden_sp = torch.tanh(
                        activation_wh_split.view(num_batch,num_type,input_split.shape[1],-1).sum(1) + torch.matmul(temp_r_sp * input_split[:,:,-self.secondary_size:], self.uh) )
                    temp_result_sp = (1 - temp_z_sp) * input_split[:,:,-self.secondary_size:] + temp_r_sp * temp_hidden_sp
                    temp_result_cb = combine_splitted_graph_output(temp_result_sp, input[:,:,-self.secondary_size:])
                    input = torch.cat((input[:,:,:-self.secondary_size], temp_result_cb), 2)
                    # input_split = torch.cat((input_split[:,:,:-self.secondary_size], temp_result_sp), 2)
            else:
                activation = torch.matmul(edge_matrix.transpose(1,2), input) + self.ba
                activation_wzrh = torch.bmm(activation, self.wz_wr_wh)
                activation_wz, activation_wr, activation_wh = torch.split(activation_wzrh, self.secondary_size, dim=-1)
                input_uzr = torch.matmul(input, self.uz_ur)
                input_uz, input_ur = torch.split(input_uzr, self.secondary_size, dim=-1)
                temp_z = torch.sigmoid(activation_wz.sum(0)+input_uz)
                temp_r = torch.sigmoid(activation_wr.sum(0)+input_ur)

                if self.secondary_size == self.size:
                    temp_hidden = torch.tanh(
                        activation_wh.sum(0) + torch.matmul(temp_r * input, self.uh))
                    input = (1 - temp_z) * input + temp_r * temp_hidden
                else:
                    temp_hidden = torch.tanh(
                        activation_wh.sum(0) + torch.matmul(temp_r * input[:,:,-self.secondary_size:], self.uh) )
                    temp_result = (1 - temp_z) * input[:,:,-self.secondary_size:] + temp_r * temp_hidden
                    input = torch.cat((input[:,:,:-self.secondary_size], temp_result), 2)

        return input

class SimpleAttention(nn.Module):
    def __init__(self, size):
        super(SimpleAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)

    def get_attention(self, x):
        attention = self.attention_net(x)
        return attention

    def forward(self, x):
        attention = self.attention_net(x)
        softmax_weight = torch.softmax(attention, dim=1)
        attention = softmax_weight * x
        sum_attention = torch.sum(attention, dim=1)
        return sum_attention

class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size/num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split, self.context_vector)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split, self.context_vector)
            softmax_weight = torch.softmax(similarity, dim=1)
            x_split = torch.cat(x.split(split_size=self.head_size, dim=2), dim=0)

            weighted_mul = torch.bmm(softmax_weight.transpose(1,2), x_split)

            restore_size = int(weighted_mul.size(0) / self.num_head)
            attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention

        