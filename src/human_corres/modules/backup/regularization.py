import os.path as osp

import torch
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.nn.inits import reset
from .pointnet2 import MLP
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, EdgeConv
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from hybrid_corres.modules import TransformationModule

class RegConv(MessagePassing):
  def __init__(self, **kwargs):
    super(RegConv, self).__init__(aggr='add', **kwargs)
    self.reset_parameters()

  def reset_parameters(self):
    pass

  def forward(self, x, pos, edge_index):
    if torch.is_tensor(pos):  # Add self-loops for symmetric adjacencies.
      edge_index, _ = remove_self_loops(edge_index)
      edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

    return self.propagate(edge_index, x=x, pos=pos)

  def message(self, x_i, x_j, pos_j, edge_index_i, size_i):
    #diff = x_i - x_j
    pred = x_i[:, :3]
    R = x_j[:, 3:12].view(-1, 3, 3)
    trans = x_j[:, 12:15].view(-1, 3, 1)
    conf_i = x_i[:, 15].view(-1, 1)
    conf_j = x_j[:, 15].view(-1, 1)
    weights = F.relu(conf_j - conf_i - 0.1)
    weights = softmax(weights, edge_index_i, size_i)
    msg = (torch.matmul(R, pos_j.unsqueeze(-1)) + trans).squeeze(-1)*weights

    return msg

  def __repr__(self):
    return '{}'.format(self.__class__.__name__)

class DistortionModule(MessagePassing):
  def __init__(self, **kwargs):
    super(DistortionModule, self).__init__(aggr='mean', **kwargs)
    self.reset_parameters()

  def reset_parameters(self):
    pass

  def forward(self, x, edge_index):
    return self.propagate(edge_index, x=x)

  def message(self, x_i, x_j):
    diff = x_i - x_j
    msg = diff.norm(p=2, dim=-1)
    return msg

  def __repr__(self):
    return '{}'.format(self.__class__.__name__)

class RegularizationModule(torch.nn.Module):
  """Our proposed Transformation Regularization module.

  """
  def __init__(self):
    super(RegularizationModule, self).__init__()
    self.conv = RegConv()

  def forward(self, x, pos, batch):
    """Compute Point-wise Rigid Transformation.
    """
    pos6d = torch.cat([pos, x[:, :3]], dim=-1)
    #idx = fps(pos6d, batch, ratio=0.2)
    #row6d, col6d = radius(pos6d, pos6d[idx], 0.05, batch, batch[idx],
    #                      max_num_neighbors=64)
    #edge_index6d = torch.stack([col6d, row6d], dim=0)
    row, col = radius(pos, pos, 0.1, batch, batch,
                      max_num_neighbors=64)
    edge_index = torch.stack([col, row], dim=0)
    new_pred = self.conv(x, pos, edge_index)
    return new_pred

class MyConv(MessagePassing):
  def __init__(self, local_nn=None, global_nn=None, **kwargs):
    super(MyConv, self).__init__(aggr='mean', **kwargs)
    self.local_nn = local_nn
    self.global_nn = global_nn
    self.reset_parameters()

  def reset_parameters(self):
    reset(self.local_nn)
    reset(self.global_nn)

  def forward(self, x, pos, edge_index):
    if torch.is_tensor(pos):  # Add self-loops for symmetric adjacencies.
      edge_index, _ = remove_self_loops(edge_index)
      edge_index, _ = add_self_loops(edge_index, num_nodes=pos.size(0))

    return self.propagate(edge_index, x=x, pos=pos)

  def message(self, x_i, x_j):
    diff = x_i - x_j
    msg = torch.cat([x_i[:, 12:], diff[:, :12].abs()], dim=1)
    if self.local_nn is not None:
      msg = self.local_nn(msg)
    return msg

  def update(self, aggr_out):
    if self.global_nn is not None:
      aggr_out = self.global_nn(aggr_out)
    return aggr_out

  def __repr__(self):
    return '{}(local_nn={}, global_nn={})'.format(
      self.__class__.__name__, self.local_nn, self.global_nn)

class PoolModule(torch.nn.Module):
  """Our proposed Transformation Regularization module.

  """
  def __init__(self, ratio, r, nn):
    super(PoolModule, self).__init__()
    self.ratio = ratio
    self.r = r
    self.conv = PointConv(nn)

  def forward(self, x, pos, batch):
    pos6d = torch.cat([x, pos], dim=-1)
    idx = fps(pos6d, batch, ratio=self.ratio)

    row6d, col6d = radius(pos6d, pos6d[idx], self.r, batch, batch[idx],
                      max_num_neighbors=64)
    edge_index6d = torch.stack([col6d, row6d], dim=0)
    x_centers = scatter_mean(x.index_select(index=col6d, dim=0),
                  row6d.unsqueeze(1), dim=0)
    #x = self.conv(x, (pos, pos[idx]), edge_index)
    pos, batch = pos[idx], batch[idx]
    return x_centers, pos, batch

class UnpoolModule(torch.nn.Module):
  """Our proposed Transformation Regularization module.

  """
  def __init__(self, k, nn):
    super(UnpoolModule, self).__init__()
    self.k = k
    self.nn = nn

  def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
    x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
    if x_skip is not None:
      x = torch.cat([x, x_skip], dim=1)
    x = self.nn(x)
    return x, pos_skip, batch_skip

#class ConfidenceModule(torch.nn.Module):
#  """Our proposed Transformation Regularization module.
#
#  """
#  def __init__(self, feat_dim, n_dim=3):
#    super(ConfidenceModule, self).__init__()
#    self.pool1 = PoolModule(0.2, 0.05, MLP([feat_dim+n_dim, 64, 64, 128]))
#    self.pool2 = PoolModule(0.3, 0.1, MLP([128+n_dim, 128, 128, 128]))
#    self.pool3 = PoolModule(0.3, 0.2, MLP([128+n_dim, 256, 256, 256]))
#    self.pool4 = PoolModule(0.3, 0.4, MLP([256+n_dim, 512, 512, 512]))
#    self.pool5 = PoolModule(0.3, 0.8, MLP([512+n_dim, 512, 512, 1024]))
#
#    self.unpool5 = UnpoolModule(1, MLP([1024+512, 512, 512]))
#    self.unpool4 = UnpoolModule(3, MLP([512+256, 256, 256]))
#    self.unpool3 = UnpoolModule(3, MLP([256+128, 256, 256]))
#    self.unpool2 = UnpoolModule(3, MLP([256+128, 256, 128]))
#    self.unpool1 = UnpoolModule(3, MLP([128+feat_dim, 128, 128, 128]))
#
#    self.lin1 = torch.nn.Linear(128, 128)
#    self.lin2 = torch.nn.Linear(128, 128)
#    self.lin3 = torch.nn.Linear(128, 1)
#
#  def forward(self, x, pos, batch):
#    pool0_out = (x, pos, batch)
#    pool1_out = self.pool1(*pool0_out)
#    pool2_out = self.pool2(*pool1_out)
#    pool3_out = self.pool3(*pool2_out)
#    pool4_out = self.pool4(*pool3_out)
#    pool5_out = self.pool5(*pool4_out)
#
#    unpool5_out = self.unpool5(*pool5_out, *pool4_out)
#    unpool4_out = self.unpool4(*unpool5_out, *pool3_out)
#    unpool3_out = self.unpool3(*unpool4_out, *pool2_out)
#    unpool2_out = self.unpool2(*unpool3_out, *pool1_out)
#    unpool1_out = self.unpool1(*unpool2_out, *pool0_out)
#
#    x = unpool1_out[0]
#
#    x = F.relu(self.lin1(x))
#    x = F.relu(self.lin2(x))
#    x = torch.sigmoid(self.lin3(x))
#
#    return x[:, 0]

class ConfEstModule(torch.nn.Module):
  def __init__(self, ratios, r6d, r3d, n_dim=3):
    super(ConfEstModule, self).__init__()
    self.ratios = ratios
    self.r3d = r3d
    self.r6d = r6d
    self.dst = DistortionModule()
    self.lin1 = torch.nn.Linear(len(self.r3d), 1)
    #self.lin2 = torch.nn.Linear(3, 1)

  def forward(self, x, pos, batch):
    pos_ext = torch.cat([pos, x], dim=1)
    msg = []
    x_centers = []
    poss = []
    idxs = []
    msg_list = []
    for r3d, ratio, r6d in zip(self.r3d, self.ratios, self.r6d):
      row6d, col6d = radius(pos_ext, pos_ext, r6d, batch, batch, max_num_neighbors=64)
      #idx = fps(pos_ext, batch, ratio=ratio)
      #idxs.append(idx)
      #row, col = radius(pos_ext, pos_ext[idx], r6d, batch, batch[idx],
      #                  max_num_neighbors=64)
      x_centers = scatter_mean(x.index_select(index=col6d, dim=0),
                     row6d.unsqueeze(1),
                     dim=0)
      edge_index6d = torch.stack([col6d, row6d], dim=0)
      #edge_indices.append(edge_index)
      row3d, col3d = radius(pos, pos, r3d,
                            batch, batch, max_num_neighbors=64)
      
      edge_index3d = torch.stack([col3d, row3d], dim=0)

      msg = self.dst(x_centers, edge_index3d)
      msg_list.append(msg.unsqueeze(-1))
    msg = torch.cat(msg_list, dim=-1)

    #conf = F.relu(self.lin1(msg))
    conf = F.sigmoid(self.lin1(msg))
    return conf[:, 0]
    #msg = torch.stack(msg, dim=-1)
    #x_centers = torch.stack(msg, dim=-1)

    #return {'pos': poss,
    #    'x_centers': x_centers,
    #    'edge_index': edge_indices,
    #    'edge_index_sub': edge_indices_sub,
    #    'idx': idxs,
    #    'msg': msg,
    #    }
