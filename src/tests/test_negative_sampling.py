import unittest
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from src.negative_sampling import structured_negative_sampling
import torch
from torch_geometric import seed_everything

class StructuredNSTest(unittest.TestCase):
    def test_type_a(self):

        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        edges = torch.LongTensor([[0, 0, 1, 1],
                                  [1, 2, 3, 4]]).cuda()
        
        tfs = torch.LongTensor([0, 1])
        targets = torch.LongTensor([1, 2, 3, 4])
        only_targets = torch.LongTensor([2, 3, 4])
        isolated = torch.LongTensor([5, 6])
        only_tfs = torch.LongTensor([1,])


        num_nodes = 7

        x = torch.rand((num_nodes, 5)).cuda()

        data = Data(edge_index=edges,
                    x=x)
        
        negative_samples = structured_negative_sampling(data, subtype="A")

        # test that we didnt sample positive edges
        self.assertEqual(edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((edges, negative_samples))).shape[1])

        from_only_targets = torch.stack([negative_samples[0] == target for target in only_targets]).sum(dim=0).bool()
        from_isolated = torch.stack([negative_samples[0] == iso for iso in isolated]).sum(dim=0).bool()
        to_isolated = torch.stack([negative_samples[1] == iso for iso in isolated]).sum(dim=0).bool()
        to_only_tfs = torch.stack([negative_samples[1] == tf for tf in only_tfs]).sum(dim=0).bool()

        self.assertTrue(from_only_targets.logical_and(to_isolated).sum() == 0)
        self.assertTrue(from_isolated.logical_and(to_isolated).sum() == 0)
        self.assertTrue(from_isolated.logical_and(to_only_tfs).sum() == 0)

    def test_type_b(self):

        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        edges = torch.LongTensor([[0, 0, 1, 1],
                                  [1, 2, 3, 4]]).cuda()
        
        tfs = torch.LongTensor([0, 1])
        targets = torch.LongTensor([1, 2, 3, 4])
        only_targets = torch.LongTensor([2, 3, 4])
        isolated = torch.LongTensor([5, 6])
        only_tfs = torch.LongTensor([1,])


        num_nodes = 7

        x = torch.rand((num_nodes, 5)).cuda()

        data = Data(edge_index=edges,
                    x=x)
        
        negative_samples = structured_negative_sampling(data, subtype="B")

        # test that we didnt sample positive edges
        self.assertEqual(edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((edges, negative_samples))).shape[1])

        from_only_targets = torch.stack([negative_samples[0] == target for target in only_targets]).sum(dim=0).bool()
        to_only_targets = torch.stack([negative_samples[1] == target for target in only_targets]).sum(dim=0).bool()
        from_isolated = torch.stack([negative_samples[0] == iso for iso in isolated]).sum(dim=0).bool()
        to_isolated = torch.stack([negative_samples[1] == iso for iso in isolated]).sum(dim=0).bool()
        to_only_tfs = torch.stack([negative_samples[1] == tf for tf in only_tfs]).sum(dim=0).bool()

        self.assertTrue(from_only_targets.logical_and(to_isolated).sum() == 0)
        self.assertTrue(from_only_targets.logical_and(to_only_targets).sum() == 0)
        self.assertTrue(from_only_targets.logical_and(to_only_tfs).sum() == 0)
        self.assertTrue(from_isolated.logical_and(to_isolated).sum() == 0)
        self.assertTrue(from_isolated.logical_and(to_only_tfs).sum() == 0)


    def test_type_c(self):
        # node 0 is only TF, node 1 is TF and target, nodes 2,3,4 are only targets, nodes 5 and 6 is isolated
        edges = torch.LongTensor([[0, 0, 1, 1],
                                  [1, 2, 3, 4]]).cuda()
        
        tfs = torch.LongTensor([0, 1])
        targets = torch.LongTensor([1, 2, 3, 4])
        only_targets = torch.LongTensor([2, 3, 4])
        isolated = torch.LongTensor([5, 6])
        only_tfs = torch.LongTensor([0,])


        num_nodes = 7

        x = torch.rand((num_nodes, 5)).cuda()

        data = Data(edge_index=edges,
                    x=x)
        
        negative_samples = structured_negative_sampling(data, subtype="C")

        # test that we didnt sample positive edges
        self.assertEqual(edges.shape[1] + negative_samples.shape[1], coalesce(torch.hstack((edges, negative_samples))).shape[1])

        from_only_targets = torch.stack([negative_samples[0] == target for target in only_targets]).sum(dim=0).bool()
        to_only_targets = torch.stack([negative_samples[1] == target for target in only_targets]).sum(dim=0).bool()
        from_isolated = torch.stack([negative_samples[0] == iso for iso in isolated]).sum(dim=0).bool()
        to_isolated = torch.stack([negative_samples[1] == iso for iso in isolated]).sum(dim=0).bool()
        from_only_tfs = torch.stack([negative_samples[0] == tf for tf in only_tfs]).sum(dim=0).bool()
        to_only_tfs = torch.stack([negative_samples[1] == tf for tf in only_tfs]).sum(dim=0).bool()

        self.assertTrue(from_only_targets.logical_and(to_isolated).sum() == 0)
        self.assertTrue(from_only_targets.logical_and(to_only_targets).sum() == 0)
        self.assertTrue(from_only_targets.logical_and(to_only_tfs).sum() == 0)
        self.assertTrue(from_only_tfs.logical_and(to_only_tfs).sum() == 0)
        self.assertTrue(from_isolated.logical_and(to_isolated).sum() == 0)
        self.assertTrue(from_isolated.logical_and(to_only_tfs).sum() == 0)



if __name__ == '__main__':
    unittest.main(warnings='ignore')

