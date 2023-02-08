import torch
from torch import nn
from torch.nn import functional as F

class MemoryModule(nn.Module):
    def __init__(self, mem_dim, fea_dim):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        # memory block initialization: N(0,1)
        self.mem_item = nn.Parameter(torch.randn((mem_dim, fea_dim)))  # (M, C)

    def feature_loss(self, query, mem_item, indices):
        query_repeat = query.repeat_interleave(self.mem_dim, dim=0)  # (B*L*M, C)
        mem_repeat = mem_item.repeat(query.shape[0], 1)  # (M*B*L, C)
        # Dij = L2(xi-mj), (i=1,2,...,L, j=1,2,...,M)
        D = torch.norm(query_repeat-mem_repeat, p=2, dim=1).reshape(-1, self.mem_dim)  # (B*L, M)
        S = nn.LogSoftmax(dim=1)(D)
        fea_loss = -torch.mean(torch.gather(S, dim=1, index=indices))
        return fea_loss

    def separation_loss(self):
        sep_loss = torch.norm(torch.triu(torch.matmul(self.mem_item,self.mem_item.T),diagonal=1))
        return sep_loss/(self.mem_dim*(self.mem_dim-1)*2)

    def forward(self, x):
        s = x.data.shape  # (B, C, L)
        x_norm = F.normalize(x, dim=1)
        x_norm = x_norm.permute(0, 2, 1).reshape(-1, s[1])  # (B*L, C)
        mem_norm = F.normalize(self.mem_item, dim=1)
        cos_sim = F.linear(x_norm, mem_norm)  # (B*L, C) X (M, C)^T = (B*L, M)
        att_weight = F.softmax(cos_sim, dim=1)  # (B*L, M)
        update = F.linear(att_weight, self.mem_item.permute(1, 0))  # (B*L, M) x (M, C) = (B*L, C)
        update = update.view(s[0], s[2], s[1])  # (B, L, C)
        update = update.permute(0, 2, 1)  # (B, C, L)
        x_query = x.permute(0, 2, 1).reshape(-1, s[1])  # (B*L, C)
        top_values, top_indices = torch.topk(att_weight, 1, dim=1)
        fea_loss = self.feature_loss(x_query, self.mem_item, top_indices)
        sep_loss = self.separation_loss()
        return update, fea_loss, sep_loss
