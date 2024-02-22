import torch


class InnerProductDecoder(torch.nn.Module):
    def __init__(self, opts):
        super(InnerProductDecoder, self).__init__()

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0, :]] * z[edge_index[1, :]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


class CosineDecoder(torch.nn.Module):
    def __init__(self, opts):
        super(CosineDecoder, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, z, edge_index, sigmoid=True):
        value = self.cos(z[edge_index[0]], z[edge_index[1]])
        return torch.sigmoid(value) if sigmoid else value


class PNormDecoder(torch.nn.Module):
    def __init__(self, opts):
        super(PNormDecoder, self).__init__()
        self.p = opts.p
        self.pnorm = torch.nn.PairwiseDistance(p=opts.p, eps=1e-06)

    def forward(self, z, edge_index, sigmoid=True):
        value = self.pnorm(z[edge_index[0]], z[edge_index[1]])
        return torch.sigmoid(value) if sigmoid else value
