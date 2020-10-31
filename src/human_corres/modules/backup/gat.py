import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, BatchNorm1d as BN
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
  def __init__(self, inp_dim, inter_dim, oup_dim, cls):
    super(GAT, self).__init__()
    self.conv1 = GATConv(inp_dim, inter_dim)
    self.bn1 = BN(inter_dim)
    self.conv2 = GATConv(inter_dim, inter_dim)
    self.bn2 = BN(inter_dim)
    self.conv3 = GATConv(inter_dim, inter_dim)
    self.bn3 = BN(inter_dim)
    self.conv4 = GATConv(inter_dim, inter_dim)
    self.bn4 = BN(inter_dim)
    #self.conv5 = GATConv(inter_dim, inter_dim)
    #self.bn5 = BN(inter_dim)
    self.lin1 = torch.nn.Linear(inter_dim, inter_dim)
    self.lin2 = torch.nn.Linear(inter_dim, inter_dim)
    self.lin3 = torch.nn.Linear(inter_dim, oup_dim)
    self.cls = cls

  def forward(self, data):
    edge_weight = None
    x1 = self.conv1(data.pos, data.edge_index, edge_weight)
    x1 = self.bn1(F.relu(x1))
    x1 = F.dropout(x1, p=0.1, training=self.training)
    x2 = self.conv2(x1, data.edge_index, edge_weight)
    x2 = self.bn2(F.relu(x2))
    x2 = F.dropout(x2, p=0.1, training=self.training)
    x3 = self.conv3(x2, data.edge_index, edge_weight)
    x3 = self.bn3(F.relu(x3))
    x3 = F.dropout(x3, p=0.1, training=self.training)
    x4 = self.conv4(x3, data.edge_index, edge_weight)
    x4 = self.bn4(F.relu(x4))
    x4 = F.dropout(x4, p=0.1, training=self.training)
    #x5 = self.conv5(x4, data.edge_index, edge_weight)
    #x5 = self.bn5(F.relu(x5))
    #x5 = F.dropout(x5, p=0.1, training=self.training)
    x = self.lin1(x4)
    x = self.lin2(F.relu(x))
    x = self.lin3(x)
    if self.cls:
      return x.log_softmax(dim=-1)
    else:
      return x
