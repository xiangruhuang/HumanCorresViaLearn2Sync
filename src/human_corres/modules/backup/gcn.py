import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, SAGPooling
from .pointnet2 import MLP

class GCN(torch.nn.Module):
  def __init__(self, inp_dim, inter_dim, oup_dim, cls=True):
    super(GCN, self).__init__()
    self.conv1 = GCNConv(inp_dim, 64)
    self.mlp1 = MLP([64, 64, 64, 128])
    self.conv2 = GCNConv(128, 128)
    self.mlp2 = MLP([128, 128, 128, 128])
    self.conv3 = GCNConv(128, 128)
    self.mlp3 = MLP([128, 256, 256, 256])
    self.conv4 = GCNConv(256, 256)
    self.mlp4 = MLP([256, 512, 512, 512])
    self.conv5 = GCNConv(512, 512)
    self.mlp5 = MLP([512, 512, 512, 1024])
    self.lin1 = torch.nn.Linear(1024, 512)
    self.lin2 = torch.nn.Linear(512, 256)
    self.lin3 = torch.nn.Linear(256, oup_dim)
    self.cls = cls

  def forward(self, data):
    edge_weight = None
    x1 = self.conv1(data.pos, data.edge_index, edge_weight)
    x1 = self.mlp1(x1)
    
    x2 = self.conv2(x1, data.edge_index, edge_weight)
    x2 = self.mlp2(x2)
    
    x3 = self.conv3(x2, data.edge_index, edge_weight)
    x3 = self.mlp3(x3)
    
    x4 = self.conv4(x3, data.edge_index, edge_weight)
    x4 = self.mlp4(x4)
    
    x5 = self.conv5(x4, data.edge_index, edge_weight)
    x5 = self.mlp5(x5)
    
    x = self.lin1(x5)
    x = self.lin2(F.relu(x))
    x = self.lin3(x)
    if self.cls:
      return x.log_softmax(dim=-1)
    else:
      return x
