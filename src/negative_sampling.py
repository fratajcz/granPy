from torch_geometric.utils import to_undirected, coalesce, remove_isolated_nodes, negative_sampling
from torch_geometric.utils import structured_negative_sampling as str_neg_sampling
import torch

def neg_sampling(data, space="full", type="structured", target=None):
    import random
    
    assert(space in ["full", "pot_net"])
    assert(type in ["tail", "head_or_tail", "random"])

    # perturb head and tail entities, sampling from full space
    if type == "head_or_tail": 
        # perturb tail node first
        result = str_neg_sampling(data.edge_index, num_nodes=data.x.shape[0],
                                  contains_neg_self_loops=False)
        
        tail_perturbed = torch.vstack((result[0], result[2]))

        # perturb head node now by switching, perturbing tail node and switching again
        result = str_neg_sampling(torch.vstack((data.edge_index[1, :], data.edge_index[0, :])), num_nodes=data.x.shape[0],
                                  contains_neg_self_loops=False)
        
        head_perturbed = torch.vstack((result[2], result[0]))

        result = torch.hstack((tail_perturbed, head_perturbed))
        
        # remove possibly sampled positive edges
        pos_edges = data.edge_index
        pos_weights = torch.ones((data.edge_index.shape[1], )).cuda()
        neg_weights = torch.zeros((result.shape[1], )).cuda()

        reduced_edges, pos_mask = coalesce(torch.hstack((result, pos_edges)), torch.cat((neg_weights, pos_weights)), reduce="add")

        result = reduced_edges[:, pos_mask == 0]

        # downsample so we have same amount as pos edges
        sample_indices = random.sample(range(result.shape[1]), data.edge_index.shape[1])
        sample_indices = torch.LongTensor(sample_indices).cuda()
        # TODO: also remove the ones going from ambivalent to ambivalent nodes
        
        return result[:, sample_indices]

    # perturb only tail node with any node
    elif type == "tail":
        result = str_neg_sampling(data.edge_index, num_nodes=data.x.shape[0],
                                              contains_neg_self_loops=False)
        result = torch.vstack((result[0], result[2]))
    
    #randomly sample edges from full space or pot_net
    elif type == "random":
        if(space == "pot_net"):
            import random
            try:
                sample_indices = random.sample(range(self.dataset.pot_net[target][0].shape[1]), data.edge_index.shape[1])
            except ValueError:
                # in case our negative set is smaller than the positive set, which can happen for test and val
                sample_indices = range(self.dataset.pot_net[target][0].shape[1])
            return self.dataset.pot_net[target][0][:, sample_indices]
    
        else:
            return negative_sampling(data.edge_index, num_nodes=data.x.shape[0])
        
    return

