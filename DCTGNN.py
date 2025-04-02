import torch
import torch.nn.functional as F
from torch import nn,einsum
from einops import rearrange, repeat
import math
from GCN_module import GCN


#==========Feature Preparation for DCTGCN==============
class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.Conv1d_block1_1 = nn.Sequential(
            nn.Conv1d(1, 16, 16, 2, 0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(16, 16),
        )

        self.Conv1d_block1_2 = nn.Sequential(
            nn.Conv1d(16, 32, 8, 1, 4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4, 0),
            nn.Flatten()
        )

        self.Conv1d_block2_1 = nn.Sequential(
            nn.Conv1d(1, 32, 64, 8, 0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
        )

        self.Conv1d_block2_2 = nn.Sequential(
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4, 0),
            nn.Flatten()
        )


    def forward(self, x):
        bs, chan, sample = x.shape
        x = x.reshape(bs * chan, 1, sample)
        x1 = self.Conv1d_block1_1(x)
        x1 = self.Conv1d_block1_2(x1)
        x2 = self.Conv1d_block2_1(x)
        x2 = self.Conv1d_block2_2(x2)
        x_out = torch.cat((x1, x2), dim=-1)
        x_out = x_out.reshape(bs, chan, -1)
        return x_out


class Position_Embedding(nn.Module):
    def __init__(self,chan_num,feat_num,tlen,T_flag,S_flag,init='xavier'):
        super(Position_Embedding, self).__init__()
        self.T_pos=None
        self.S_pos=None
        self.init_flag = init
        self.N=chan_num
        self.T=tlen
        if T_flag:
            self.T_pos = nn.Parameter(torch.randn(self.T, feat_num))
            self.pos_init(self.T_pos)

        if S_flag:
            self.S_pos = nn.Parameter(torch.randn(self.N, feat_num))
            self.pos_init(self.S_pos)

    def forward(self, data):
        if self.T_pos is not None:
            T_pos = self.T_pos.unsqueeze(1)
            T_pos = T_pos.repeat(1, self.N, 1)
            data = data + T_pos
        if self.S_pos is not None:
            S_pos = self.S_pos.unsqueeze(0)
            S_pos = S_pos.repeat(self.T, 1, 1)
            data = data + S_pos
        return data

    def pos_init(self,x):
        if self.init_flag=='xavier':
            nn.init.xavier_uniform_(x)
        if self.init_flag=='kaiming':
            nn.init.kaiming_uniform_(x)
#==============================================


#==========DCTGCN-related Modules==============

class AdjGenerator(nn.Module):
    def __init__(self, device, t_win, dim, Tem=1, init='xavier'):
        super().__init__()
        self.scale = Tem ** -1
        self.device = device
        self.twin = t_win
        self.Softmax = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, dim//4, bias=False)
        self.to_k = nn.Linear(dim, dim//4, bias=False)
        if init == 'xavier':
            self._para_init()

    def _para_init(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.Softmax(dots)
        return attn


class N2N(nn.Module):
    def __init__(self, device, in_dim, win_size,init_qk,K_GCN,output_dim ):
        super(N2N, self).__init__()
        self.E_A_gene=AdjGenerator(device, win_size, in_dim, 1, init_qk)
        self.E_GCN=GCN(in_dim, output_dim, K_GCN)

    def forward(self, x):
        E_A_input = self.E_A_gene(x)
        out = self.E_GCN(x, E_A_input)
        return out,E_A_input

class N2R_R2R_R2N(nn.Module):
    def __init__(self, device, in_dim, reg_dim, win_size, n_chan, dataset_name, opt_idx, K_R_GCN, init_qk):
        super().__init__()
        self.device = device
        self.twin = win_size
        self.n_chan = n_chan
        self.device = device
        self.Q = nn.Parameter(torch.randn(n_chan * win_size, in_dim))
        self.R_GCN = GCN(in_dim, reg_dim, K_R_GCN)
        self.P = self.region_select(dataset_name, opt_idx)
        self.R_A_gene = AdjGenerator(device, self.twin, in_dim, 1,init_qk)

    def region_select(self, dataset_name, opt_idx):
        Reg2Chan = None
        if dataset_name == 'MODMA':

            if opt_idx == 1:
                # R1:FP,R2:F,R3:C,R4:P,R5:O,R6:T-left(T-l),R7:T-right(T-r)
                Reg2Chan = {
                    'R1': [0, 1],
                    'R2': [3, 4, 2, 5, 6],
                    'R3': [8, 9],
                    'R4': [ 12, 13, 14],
                    'R5': [16, 17, 18],
                    'R6': [7,11],
                    'R7': [10,15]
                }

            elif opt_idx == 2:
                # R1:F-l,R2:F-r,R3:FCP,R4:CPO-l,R5:CPO-r,R6:T-l,R7:T-r
                Reg2Chan = {
                    'R1': [0, 3, 4],
                    'R2': [1, 5, 6],
                    'R3': [2, 13, 17],
                    'R4': [8, 12, 16],
                    'R5': [9,  14, 18],
                    'R6': [7, 11],
                    'R7': [10, 15]
                }

            elif opt_idx == 3:
                # R1:F-l,R2:F-r,R3:C,R4:P,R5:O,R6:T-l,R7:T-r
                Reg2Chan = {
                    'R1': [0, 3, 4],
                    'R2': [1, 5, 6],
                    'R3': [ 8, 2, 9],
                    'R4': [ 12, 13, 14],
                    'R5': [16, 17, 18],
                    'R6': [7, 11],
                    'R7': [10, 15]
                }

        elif dataset_name == 'HUSM':
            if opt_idx == 1:
                Reg2Chan = {
                    'R1': [0, 9],
                    'R2': [5, 1, 8, 10, 14],
                    'R3': [ 2, 17, 11],
                    'R4': [ 3, 18, 12],
                    'R5': [4, 13],
                    'R6': [6, 7],
                    'R7': [15, 16],

                }
            elif opt_idx == 2:
                Reg2Chan = {
                    'R1': [0, 5, 1],
                    'R2': [9, 10, 14],
                    'R3': [8, 17, 18],
                    'R4': [ 2, 3, 4],
                    'R5': [11, 12, 13],
                    'R6': [6, 7],
                    'R7': [15, 16],
                }
            elif opt_idx == 3:
                Reg2Chan = {
                    'R1': [0, 5, 1],
                    'R2': [9, 10, 14],
                    'R3': [ 2, 17, 8, 11],
                    'R4': [ 3, 18, 12],
                    'R5': [4, 13],
                    'R6': [6, 7],
                    'R7': [15, 16],
                }


        P = torch.zeros((7, 19), device=self.device)
        for i, indices in enumerate(Reg2Chan.values()):
            P[i, indices] = 1
        row_sums = P.sum(dim=1, keepdim=True)
        P = P / row_sums
        P_expand = torch.zeros((7 * self.twin, self.n_chan * self.twin), device=self.device)
        for i in range(self.twin):
            P_expand[i * 7:(i + 1) * 7, i * self.n_chan:(i + 1) * self.n_chan] = P
        return P_expand

    def forward(self, x):
        bs, _, _ = x.shape
        P = self.P.unsqueeze(0).repeat(bs, 1, 1)
        Q = self.Q.unsqueeze(0).repeat(bs, 1, 1)
        x_filtered = F.relu(Q * x)
        x_reg = torch.matmul(P, x_filtered)
        A_reg = self.R_A_gene(x_reg)
        x_reg = self.R_GCN(x_reg, A_reg)
        reg2node = torch.matmul(P.transpose(1, 2), x_reg)
        return reg2node, A_reg




def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = torch.transpose(input, 1, 3)

## 这是一个二维的操作，需要的是二维的kernel，所以这里是(num_sensors, time_window_size)
    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride)

    y_ = torch.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = torch.transpose(y_, 1, -1)

    return y_


class DCTGCN(nn.Module):
    def __init__(self, device, chan, input_dim, output_dim, time_window_size, stride, init_qk, K_GCN, K_R_GCN, dn, P_s):
        super(DCTGCN, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim
        self.BN = nn.BatchNorm1d(input_dim)

        self.Reg_Level = N2R_R2R_R2N(device=device, in_dim=input_dim, reg_dim=output_dim, win_size=time_window_size,
                                     n_chan=chan, dataset_name=dn, opt_idx=P_s, K_R_GCN=K_R_GCN, init_qk=init_qk)

        self.Electrode_Level = N2N(device=device, in_dim=input_dim, win_size=time_window_size, init_qk=init_qk,
                                   K_GCN=K_GCN, output_dim=output_dim)


    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = torch.transpose(input_con, 2, 3)
        input_con_ = torch.reshape(input_con_, [bs * num_windows, time_window_size * num_sensors, feature_dim])
        input_con_ = torch.transpose(input_con_, -1, -2)
        input_con_ = self.BN(input_con_)
        input_con_ = torch.transpose(input_con_, -1, -2)

        X_output_N,A_E = self.Electrode_Level(input_con_)
        X_output_N = torch.reshape(X_output_N, [bs, num_windows, time_window_size, num_sensors, self.output_dim])

        X_output_R,A_R = self.Reg_Level(input_con_)
        X_output_R = torch.reshape(X_output_R, [bs, num_windows, time_window_size, num_sensors, self.output_dim])

        X_output = X_output_N + X_output_R
        X_output = torch.mean(X_output, 2)

        return X_output, A_E, A_R

#==========================================================



#==========Attention-based Fusion Layer (AFL)==============
class AFL(nn.Module):
    def __init__(self, in_dim, hid_r):
        super(AFL, self).__init__()
        self.W_b = nn.Linear(in_dim, in_dim//hid_r, bias=True)
        self.q = nn.Linear(in_dim//hid_r, 1, bias=False)

    def forward(self, z):
        bs, nw, chan, dim = z.shape
        a = self.q(F.tanh(self.W_b(z))).squeeze(-1)
        a = F.softmax(a, dim=-2)
        a = a.unsqueeze(3).repeat(1, 1, 1, dim)
        weighted_z = z * a
        atten_z = torch.sum(weighted_z, dim=1)
        return atten_z

#=========================================================






class DCTGNN(nn.Module):
    def __init__(self, device, tlen,time_window1, time_window2,stride1,stride2, feat_emb, chan_num, gnn_out_chan, atten_hid_r,atten_opt,K_GCN,K_R_GCN,P_s,dn,
                 T_flag=True,S_flag=False,init_method='randn',init_qk='xavier'):
        '''
        :param device: gpu or cpu
        :param tlen:  Number of Time steps. Set as four in our experiments
        :param time_window1: Length of sliding window1. Set as two in our experiments
        :param time_window2: Length of sliding window2. Set as three in our experiments
        :param stride1: Stride of sliding window1. Set as one in our experiments
        :param stride2: Stride of sliding window2. Set as one in our experiments
        :param feat_emb: Dim of the preliminary features encoded by 1DCNN
        :param chan_num: Number of EEG electrodes
        :param gnn_out_chan: Dim of the output of GNN
        :param atten_hid_r: hid_r in AFL
        :param atten_opt: whether to adopt AFL
        :param K_GCN: Layers of GCN in E2E Branch
        :param K_R_GCN: Layers of GCN in R2R Branch
        :param P_s: Partition_Scheme for E2R (Electrode-to-Region) 0=general 1=hemisphere 2=frontal
        :param dn: dataset_name  'MODMA' or 'HUSM'
        :param T_flag: True for position embedding in Temporal dimension
        :param S_flag: position embedding in Spatial dimension (Set as False in our experiment)
        :param init_method: init method for Position_Embedding
        :param init_qk: init method for qk in self-attention
        '''

        super(DCTGNN, self).__init__()
        self.Feature = FeatureEncoder()
        self.atten_opt = atten_opt
        self.st_pos = Position_Embedding(chan_num, feat_emb, tlen, T_flag, S_flag, init_method)
        self.gnn1 = DCTGCN(device=device, chan=chan_num, input_dim=feat_emb, output_dim=gnn_out_chan, time_window_size=time_window1, stride=stride1,init_qk=init_qk,K_GCN=K_GCN,K_R_GCN=K_R_GCN,P_s=P_s,dn=dn)
        self.gnn2 = DCTGCN(device=device, chan=chan_num, input_dim=feat_emb, output_dim=gnn_out_chan, time_window_size=time_window2, stride=stride2,init_qk=init_qk,K_GCN=K_GCN,K_R_GCN=K_R_GCN,P_s=P_s,dn=dn)
        self.atten_sum = AFL(gnn_out_chan, atten_hid_r)
        self.fc = nn.Linear(gnn_out_chan * chan_num, 2)

    def forward(self, x):
        bs, channum, tlen, sp = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(-1, channum, sp)
        x = self.Feature(x)
        x = x.reshape(bs, tlen, channum, -1)

        x = x + self.st_pos(x)

        feat1, A1,RA1 = self.gnn1(x)
        feat2, A2,RA2 = self.gnn2(x)
        feat = torch.concat((feat1, feat2), dim=1)


        if self.atten_opt:
            feat = self.atten_sum(feat)
        else:
            feat = torch.sum(feat, dim=1)

        feat = feat.reshape(bs, -1)

        out = self.fc(feat)

        return feat, out, A1, A2, RA1, RA2


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.rand((128, 19, 4, 250)).cuda()
    net = DCTGNN(device, tlen=4, time_window1=2, time_window2=3, stride1=1, stride2=1,
                  feat_emb=128, chan_num=19, gnn_out_chan=32, atten_hid_r=16,
                  atten_opt=True, K_GCN=2, K_R_GCN=2, P_s=2, dn='MODMA').cuda()

    _, out, A1, A2, RA1, RA2 = net(data)
    print(f'Shape of output: {out.shape}')
    print(f'Shape of A in Sliding Window1: {A1.shape}')
    print(f'Shape of RA in Sliding Window1: {RA1.shape}')
    print(f'Shape of A in Sliding Window2: {A2.shape}')
    print(f'Shape of RA in Sliding Window2: {RA2.shape}')



