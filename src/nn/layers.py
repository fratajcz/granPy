import torch_geometric.nn as pyg_nn
import torch
from torch_geometric.utils import degree, add_remaining_self_loops


class NoneConv(pyg_nn.GCNConv):
    """ Behaves just as pyg_nn.GCNConv but the aggregation function does nothing, i.e. each node completely retains its identity """
    def forward(self, x, edge_index):

        edge_index = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])[0]

        out_degree = degree(edge_index[0, :])
        in_degree = degree(edge_index[1, :])

        x *= torch.sqrt(in_degree).reshape((-1, 1))
        x *= torch.sqrt(out_degree).reshape((-1, 1))

        out = self.lin(x)

        if self.bias is not None:
            out = out + self.bias

        return out
