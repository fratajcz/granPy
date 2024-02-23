from torch_geometric.utils import to_undirected, coalesce, remove_isolated_nodes
from torch_geometric.utils import structured_negative_sampling as str_neg_sampling
import torch

def structured_negative_sampling(data, subtype="A"):
    import random
    
    if subtype not in ["A", "B", "C"]:
        raise ValueError("Structured negative sampling only has subtypes A, B and C.")
    
    isolated_nodes = (~remove_isolated_nodes(data.edge_index, num_nodes=data.x.shape[0])[2]).nonzero()
    tfs = torch.unique(data.edge_index[0, :]).long()
    targets = torch.unique(data.edge_index[1, :]).long()

    # make edge index undirected, meaning we perturb head and tail entities, sampling from all nodes
    if subtype == "A":
        sampling_space = torch.hstack((data.edge_index, torch.vstack((data.edge_index[1, :], data.edge_index[0, :]))))  # perturb head or tail node
        result = str_neg_sampling(sampling_space, num_nodes=data.x.shape[0] - 1,
                                  contains_neg_self_loops=False)
        # need to remove edges from targets to isolated bc they would require two perturbations
        from_target = torch.cat((torch.zeros(data.edge_index.shape[1],), torch.ones((data.edge_index.shape[1], )))).bool().cuda()
        to_isolated = torch.stack([result[2] == isolated for isolated in isolated_nodes]).sum(dim=0).bool()

        mask = ~from_target.logical_and(to_isolated)

        result = torch.vstack((result[0][mask], result[2][mask]))

    if subtype == "B":
        sampling_space = torch.hstack((data.edge_index, data.edge_index))  # perturb only tail node with any node
        result = str_neg_sampling(sampling_space, num_nodes=data.x.shape[0],
                                              contains_neg_self_loops=False)

        result = torch.vstack((result[0], result[2]))
    
    if subtype == "C":
        sampling_space = torch.hstack((data.edge_index, data.edge_index))  # perturb only tail node with any node
        result = str_neg_sampling(sampling_space, num_nodes=data.x.shape[0],
                                              contains_neg_self_loops=False)
    
        only_tfs = [tf for tf in tfs if tf not in targets]
        to_only_tf = torch.stack([result[2] == tf for tf in only_tfs]).sum(dim=0).bool()

        mask = ~to_only_tf

        result = torch.vstack((result[0][mask], result[2][mask]))
            
    

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

