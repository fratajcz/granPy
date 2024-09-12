import unittest
import src.nn.layers as own_layers
import torch


class NoneConvTest(unittest.TestCase):

    def test_forward(self):
        edges1 = torch.LongTensor([[0, 1, 2, 3],
                                   [1, 2, 3, 0]])
        
        x = torch.rand((4, 4))

        conv = own_layers.NoneConv(4, 4)

        conv(x, edges1)

        self.assertTrue(True)

    def test_same_degree_equal_output(self):
        edges1 = torch.LongTensor([[0, 1, 2, 3],
                                   [1, 2, 3, 0]])
        
        edges2 = torch.LongTensor([[0, 1, 2, 3],
                                  [2, 3, 1, 0]])  # here we have (0,2) instead of (0,1) and (2,1) insteead of (2,3)) but same degree for all
        
        x = torch.rand((4, 4))

        conv = own_layers.NoneConv(4, 4)

        out1 = conv(x, edges1)
        out2 = conv(x, edges2)
        print(out1)
        print(out2)
        self.assertTrue(torch.eq(out1, out2).all())

    def test_different_degree_different_output(self):
        edges1 = torch.LongTensor([[0, 1, 2, 3],
                                   [1, 2, 3, 0]])
        
        edges2 = torch.LongTensor([[0, 1, 2, 3],
                                  [2, 3, 3, 3]])  # here we have (0,2) instead of (0,1) and (2,1) insteead of (2,3)) but same degree for all
        
        x = torch.rand((4, 4))

        conv = own_layers.NoneConv(4, 4)

        out1 = conv(x, edges1)
        out2 = conv(x, edges2)

        self.assertFalse(torch.eq(out1, out2).all())
