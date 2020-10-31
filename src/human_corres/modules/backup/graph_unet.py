import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GraphUNet as UNet
from torch_geometric.utils import dropout_adj

class GraphUNet(torch.nn.Module):
  def __init__(self, inp_dim, inter_dim, oup_dim, cls):
    super(GraphUNet, self).__init__()
    pool_ratios = [2000 / 5000.0, 0.5]
    self.unet = UNet(inp_dim, inter_dim, inter_dim,
             depth=3, pool_ratios=pool_ratios)
    self.lin1 = torch.nn.Linear(inter_dim, inter_dim)
    self.lin2 = torch.nn.Linear(inter_dim, inter_dim)
    self.lin3 = torch.nn.Linear(inter_dim, oup_dim)
    self.cls = cls

  def forward(self, data):
    edge_index, _ = dropout_adj(data.edge_index, p=0.1,
                  force_undirected=True,
                  num_nodes=data.num_nodes,
                  training=self.training)
    x = data.pos #F.dropout(data.pos, p=0.1, training=self.training)
    x = self.unet(x, edge_index)
    x = self.lin1(x)
    x = self.lin2(F.relu(x))
    x = self.lin3(x)
    if self.cls:
      return F.log_softmax(x, dim=1)
    else:
      return x

