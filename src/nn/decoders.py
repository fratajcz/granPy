import torch
from torch_geometric.utils import degree


class InnerProductDecoder(torch.nn.Module):
    def __init__(self, opts):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z, edge_index, sigmoid=True, *args, **kwargs):
        value = (z[edge_index[0, :]] * z[edge_index[1, :]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class CosineDecoder(torch.nn.Module):
    def __init__(self, opts):
        super(CosineDecoder, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, z, edge_index, sigmoid=True, *args, **kwargs):
        value = self.cos(z[edge_index[0]], z[edge_index[1]])
        return torch.sigmoid(value) if sigmoid else value


class PNormDecoder(torch.nn.Module):
    def __init__(self, opts):
        super(PNormDecoder, self).__init__()
        self.p = opts.p
        self.pnorm = torch.nn.PairwiseDistance(p=opts.p, eps=1e-06)

    def forward(self, z, edge_index, sigmoid=True, *args, **kwargs):
        value = self.pnorm(z[edge_index[0]], z[edge_index[1]])
        return torch.sigmoid(value) if sigmoid else value


class DegreeSorter(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z, edge_index, pos_edge_index):
        tail_nodes = edge_index[1, :]
        degrees = degree(pos_edge_index[1, :], num_nodes=z.shape[0])
        return degrees[tail_nodes]


class OutDegreeSorter(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z, edge_index, pos_edge_index):
        head_nodes = edge_index[0, :]
        degrees = degree(pos_edge_index[0, :], num_nodes=z.shape[0])
        return degrees[head_nodes]


class SymmetricDegreeSorter(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z, edge_index, pos_edge_index):
        tail_nodes = edge_index[1, :]
        head_nodes = edge_index[0, :]
        in_degrees = degree(pos_edge_index[1, :], num_nodes=z.shape[0])
        out_degrees = degree(pos_edge_index[0, :], num_nodes=z.shape[0])
        return (in_degrees[tail_nodes] + out_degrees[head_nodes]) / 2


class HarmonicDegreeSorter(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, z, edge_index, pos_edge_index, eps=1e-16):
        tail_nodes = edge_index[1, :]
        head_nodes = edge_index[0, :]
        in_degrees = degree(pos_edge_index[1, :], num_nodes=z.shape[0]) + eps
        out_degrees = degree(pos_edge_index[0, :], num_nodes=z.shape[0]) + eps
        return 2 / (in_degrees[tail_nodes].pow(-1) + out_degrees[head_nodes].pow(-1))
