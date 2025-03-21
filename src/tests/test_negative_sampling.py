import unittest
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from src.negative_sampling import neg_sampling
from src.datasets.datasets import GranPyDataset
import torch

class StructuredNSTest(unittest.TestCase):
    def test_structured_tail(self):

        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        train_edges = torch.LongTensor([[0, 0, 1, 1],
                                        [1, 2, 3, 4]])
        val_edges = torch.LongTensor([[0, 1],
                                      [3, 2]])
        
        num_nodes = 7
        not_tfs = torch.LongTensor([2, 3, 4, 5, 6])

        train_data = Data(pos_edges=train_edges, known_edges=train_edges, num_nodes = num_nodes, known_edges_label = torch.ones(train_edges.shape[1]))
        val_data =  Data(pos_edges=val_edges, num_nodes=num_nodes)
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        val_data.known_edges_label = torch.hstack((1-train_data.known_edges_label, torch.ones((val_data.pos_edges.shape[1],))))
        
        negative_samples = neg_sampling(val_data, space="pot_net", type="tail")

        # test that we didnt sample positive edges or duplicate negatives
        self.assertEqual(val_data.known_edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((val_data.known_edges, negative_samples))).shape[1])
        
        # test that we sampled the right number of edges
        self.assertEqual(negative_samples.shape[1], val_data.pos_edges.shape[1])

        from_not_tfs = torch.stack([negative_samples[0] == target for target in not_tfs]).sum(dim=0).bool()
        self.assertTrue(from_not_tfs.sum() == 0)


    def test_structured_head_or_tail(self):

        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        train_edges = torch.LongTensor([[0, 0, 1, 1],
                                        [1, 2, 3, 4]])
        val_edges = torch.LongTensor([[0, 1],
                                      [3, 2]])
        
        num_nodes = 7
        only_tfs = torch.LongTensor([0])
        only_targets = torch.LongTensor([2, 3, 4])
        iso = torch.LongTensor([5, 6])

        train_data = Data(pos_edges=train_edges, known_edges=train_edges, num_nodes = num_nodes,known_edges_label = torch.ones(train_edges.shape[1]))
        val_data =  Data(pos_edges=val_edges, num_nodes=num_nodes)
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        val_data.known_edges_label = torch.hstack((1-train_data.known_edges_label, torch.ones((val_data.pos_edges.shape[1],))))
        
        negative_samples = neg_sampling(val_data, space="pot_net", type="head_or_tail")

        # test that we didnt sample positive edges or duplicate negatives
        self.assertEqual(val_data.known_edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((val_data.known_edges, negative_samples))).shape[1])
        
        # test that we sampled the right number of edges
        self.assertEqual(negative_samples.shape[1], val_data.pos_edges.shape[1])

        from_only_targets = torch.stack([negative_samples[0] == target for target in only_targets]).sum(dim=0).bool()
        from_iso = torch.stack([negative_samples[0] == target for target in iso]).sum(dim=0).bool()
        to_iso = torch.stack([negative_samples[1] == target for target in iso]).sum(dim=0).bool()
        to_only_tf = torch.stack([negative_samples[1] == target for target in only_tfs]).sum(dim=0).bool()
        
        self.assertTrue(from_iso.logical_and(to_iso).sum() == 0)
        self.assertTrue(from_iso.logical_and(to_only_tf).sum() == 0)
        self.assertTrue(from_only_targets.logical_and(to_iso).sum() == 0)
        self.assertTrue(from_only_targets.logical_and(to_only_tf).sum() == 0)

    def test_pot_net(self):
        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        train_edges = torch.LongTensor([[0, 0, 1, 1],
                                        [1, 2, 3, 4]])
        val_edges = torch.LongTensor([[0, 1],
                                      [3, 2]])
        
        num_nodes = 7
        not_tfs = torch.LongTensor([2, 3, 4, 5, 6])

        train_data = Data(pos_edges=train_edges, known_edges=train_edges, num_nodes = num_nodes, known_edges_label = torch.ones(train_edges.shape[1]))
        val_data =  Data(pos_edges=val_edges, num_nodes=num_nodes)
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        val_data.known_edges_label = torch.hstack((1-train_data.known_edges_label, torch.ones((val_data.pos_edges.shape[1],))))
        val_data.pot_net = GranPyDataset.construct_pot_net(val_data)
        
        negative_samples = neg_sampling(val_data, space="pot_net", type="random")

        # test that we didnt sample positive edges or duplicate negatives
        self.assertEqual(val_data.known_edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((val_data.known_edges, negative_samples))).shape[1])
        
        # test that we sampled the right number of edges
        self.assertEqual(negative_samples.shape[1], val_data.pos_edges.shape[1])

        from_not_tfs = torch.stack([negative_samples[0] == target for target in not_tfs]).sum(dim=0).bool()
        self.assertTrue(from_not_tfs.sum() == 0)
        
    def test_random(self):
        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        train_edges = torch.LongTensor([[0, 0, 1, 1],
                                        [1, 2, 3, 4]])
        val_edges = torch.LongTensor([[0, 1],
                                      [3, 2]])
        
        num_nodes = 7

        train_data = Data(pos_edges=train_edges, known_edges=train_edges, num_nodes = num_nodes, known_edges_label = torch.ones(train_edges.shape[1]))
        val_data =  Data(pos_edges=val_edges, num_nodes=num_nodes)
        val_data.known_edges = torch.hstack((train_data.known_edges, val_data.pos_edges))
        
        negative_samples = neg_sampling(val_data, space="full", type="random")

        # test that we didnt sample positive edges or duplicate negatives
        self.assertEqual(val_data.known_edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((val_data.known_edges, negative_samples))).shape[1])
        
        # test that we sampled the right number of edges
        self.assertEqual(negative_samples.shape[1], val_data.pos_edges.shape[1])


if __name__ == '__main__':
    unittest.main(warnings='ignore')

