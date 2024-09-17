from torch_geometric.utils import negative_sampling, coalesce
from torch_geometric.utils import structured_negative_sampling as str_neg_sampling
from torch_geometric.utils import structured_negative_sampling_feasible as str_negative_sampling_feasible
import torch

def neg_sampling(data, space="full", type="tail"):
    import random
    
    assert(space in ["full", "pot_net"])
    assert(type in ["tail", "head_or_tail", "random"])
    
    #randomly sample edges from full space or pot_net
    if type == "random":
        if(space == "pot_net"):
            try:
                sample_indices = random.sample(range(data.pot_net.shape[1]), data.pos_edges.shape[1])
            except ValueError:
                # in case our negative set is smaller than the positive set, which can happen for test and val
                sample_indices = range(data.pot_net.shape[1])
            return data.pot_net[:, sample_indices]
        else:
            return negative_sampling(data.known_edges, num_nodes=data.num_nodes, num_neg_samples=data.pos_edges.shape[1]) 

    elif type in ["tail", "head_or_tail"]:
        assert(str_negative_sampling_feasible(data.known_edges, num_nodes=data.num_nodes,
                                  contains_neg_self_loops=False))
        
        # perturb tail node first
        result = str_neg_sampling(data.known_edges, num_nodes=data.num_nodes,
                                  contains_neg_self_loops=False)
        
        tail_perturbed = torch.vstack((result[0][data.known_edges_label == 1], result[2][data.known_edges_label == 1]))
        
        if(type == "tail"): 
            return tail_perturbed
        
        # perturb head node now by switching, perturbing tail node and switching again
        inv_edges = torch.vstack((data.known_edges[1, :], data.known_edges[0, :]))
        
        assert(str_negative_sampling_feasible(inv_edges, num_nodes=data.num_nodes, contains_neg_self_loops=False))
        result = str_neg_sampling(inv_edges, num_nodes=data.num_nodes, contains_neg_self_loops=False)
        
        head_perturbed = torch.vstack((result[2][data.known_edges_label == 1], result[0][data.known_edges_label == 1]))

        result = coalesce(torch.hstack((tail_perturbed, head_perturbed)))

        # downsample so we have same amount as pos edges
        sample_indices = random.sample(range(result.shape[1]), data.pos_edges.shape[1])
        sample_indices = torch.LongTensor(sample_indices)
        
        return result[:, sample_indices]
    
    
