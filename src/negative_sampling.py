from torch_geometric.utils import to_undirected, coalesce, remove_isolated_nodes, negative_sampling
from torch_geometric.utils import structured_negative_sampling as str_neg_sampling
from torch_geometric.utils import structured_negative_sampling_feasible as str_negative_sampling_feasible
import torch

def neg_sampling(data, space="full", type="structured", target=None):
    import random
    
    assert(space in ["full", "pot_net"])
    assert(type in ["tail", "head_or_tail", "random"])
    
    #randomly sample edges from full space or pot_net
    if type == "random":
        if(space == "pot_net"):
            try:
                sample_indices = random.sample(range(data.pot_net[0].shape[1]), data.edge_index.shape[1])
            except ValueError:
                # in case our negative set is smaller than the positive set, which can happen for test and val
                sample_indices = range(data.pot_net[0].shape[1])
            return data.pot_net[0][:, sample_indices]
        else:
            return negative_sampling(data.edge_index, num_nodes=data.x.shape[0])

    elif type in ["tail", "head_or_tail"]:
        assert(str_negative_sampling_feasible(data.known_edges, num_nodes=data.x.shape[0],
                                  contains_neg_self_loops=False))
        
        # perturb tail node first
        result = str_neg_sampling(data.known_edges, num_nodes=data.x.shape[0],
                                  contains_neg_self_loops=False)
        assert(result[0].shape[1] == data.known_edges.shape[1])
        
        tail_perturbed = torch.vstack((result[0][data.known_edges_label == 1], result[2][data.known_edges_label == 1]))
        
        if(type == "tail"): 
            return tail_perturbed
        
        # perturb head node now by switching, perturbing tail node and switching again
        assert(str_negative_sampling_feasible(torch.vstack((data.known_edges[1, :], data.known_edges[0, :])), num_nodes=data.x.shape[0],
                                  contains_neg_self_loops=False))
               
        inv_edges = torch.vstack((data.known_edges[1, :], data.known_edges[0, :]))
        edge_index = torch.hstack((data.known_edges, inv_edges))
        edge_index_label = torch.hstack((torch.zeros((data.known_edges.shape[1],)), torch.ones((inv_edges.shape[1],))))
        
        assert(str_negative_sampling_feasible(edge_index, num_nodes=data.x.shape[0], contains_neg_self_loops=False))
        result = str_neg_sampling(edge_index, num_nodes=data.x.shape[0], contains_neg_self_loops=False)
        assert(result[0].shape[1] == data.known_edges.shape[1])
        
        head_perturbed = torch.vstack((result[2][edge_index_label == 1], result[0][edge_index_label == 1]))

        result = torch.hstack((tail_perturbed, head_perturbed))

        # downsample so we have same amount as pos edges
        sample_indices = random.sample(range(result.shape[1]), data.edge_index.shape[1])
        sample_indices = torch.LongTensor(sample_indices).cuda()
        
        return result[:, sample_indices]
    
    
