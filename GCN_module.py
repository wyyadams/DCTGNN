import torch
import torch.nn.functional as F
from torch import nn,einsum
import math

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False, init='x'):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.init = init
        self._para_init()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def _para_init(self):
        if self.init == 'xavier':
            nn.init.xavier_normal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class GNNnet(nn.Module):
    def __init__(self,in_channels, out_channels, K):
        super(GNNnet, self).__init__()
        self.K = K
        self.gnn = nn.ModuleList()
        for i in range(self.K):
            self.gnn.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x, L):
        adj = self.generate_adj(L, self.K)
        for i in range(len(self.gnn)):
            if i == 0:
                result = F.leaky_relu(self.gnn[i](x, adj[i]))
            else:
                result += F.leaky_relu(self.gnn[i](x, adj[i]))
        return result

    def generate_adj(self, L, K):
        support = []
        L_iter = L
        for i in range(K):
            if i == 0:
                support.append(torch.eye(L.shape[-1]).cuda())
            else:
                support.append(L_iter)
                L_iter = L_iter * L
        return support


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.GCN = GNNnet(in_dim, out_dim, K)

    def forward(self, x, A):
        L = self.BatchAdjNorm(A)
        out = self.GCN(x, L)

        return out


    def BatchAdjNorm(self, A):
        bs, n, _ = A.shape
        A = F.relu(A)
        identity = torch.eye(n, n, device=A.device)
        identity_matrix = identity.repeat(bs, 1, 1)
        A = A * (torch.ones(bs, n, n, device=A.device) - identity_matrix)
        A = A + identity_matrix
        d = torch.sum(A, 2)
        d = 1 / torch.sqrt((d + 1e-10))
        D = torch.diag_embed(d)
        Lnorm = torch.matmul(torch.matmul(D, A), D)
        return Lnorm