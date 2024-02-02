import torch_geometric.nn as pyg_nn
from torch_geometric.typing import (
    Adj
)
import torch
from torch_geometric.utils import degree

class NoneConv(pyg_nn.GCNConv):
    """ Behaves just as pyg_nn.GCNConv butthe aggregation function does nothing, i.e. each node completely retains its identity """
    def forward(self, x, edge_index):

        out_degree = degree(edge_index[0, :])
        in_degree = degree(edge_index[1, :])

        x *= torch.sqrt(in_degree).reshape((-1, 1))
        x *= torch.sqrt(out_degree).reshape((-1, 1))

        out = self.lin(x)

        if self.bias is not None:
            out = out + self.bias

        return out
