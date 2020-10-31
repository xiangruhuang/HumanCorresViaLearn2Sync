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
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, EdgeConv, knn
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from hybrid_corres.modules import TransformationModule
from hybrid_corres.utils import rigid_fitting
import numpy as np

class PoolingModule(torch.nn.Module):
  def __init__(self, ratio, r6d, **kwargs):
    super(PoolingModule, self).__init__(**kwargs)
    self.ratio = ratio
    self.r6d = r6d

  def forward(self, x, pos, batch):
    pos6d = torch.cat([pos, x], dim=-1)
    idx = fps(pos6d, batch, ratio=self.ratio, random_start=False)
    #batch0_mask = (batch[idx] == 0)
    #print('idx^2={}'.format((idx[batch0_mask]**2).sum()))
    row, col = radius(pos6d, pos6d[idx], self.r6d,
                        batch, batch[idx], max_num_neighbors=128)
    edge_index = torch.stack([col, row], dim=0)
    #knn_index = knn(pos6d[idx], pos6d, 1, batch[idx], batch)
    x = scatter_mean(
          x.index_select(index=edge_index[0], dim=0),
          edge_index[1].unsqueeze(1),
          dim=0)
    pos = scatter_mean(
            pos.index_select(index=edge_index[0], dim=0),
            edge_index[1].unsqueeze(1),
            dim=0)

    return (x, pos, batch[idx]), edge_index

class CDFConv(MessagePassing):
  def __init__(self, aggr, r_min, r_max, num_cdfsamples, **kwargs):
    super(CDFConv, self).__init__(aggr=aggr, **kwargs)
    thresholds = np.linspace(r_min, r_max, num_cdfsamples)
    thresholds = torch.Tensor(thresholds).view(1, num_cdfsamples)
    self.register_buffer('thresholds', thresholds)
    #self.lin1 = torch.nn.Linear(1, oup_dim)

    #torch.nn.init.constant_(self.lin1.bias, torch.ones_like(self.lin1.bias))
    #torch.nn.init.constant_(self.lin1.weights, torch.ones_like(self.lin1.weights)*(-1))
    #self.bias = torch.Tensor([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]).view(1, -1)
    #self.w1 = torch.Tensor([100.0])

  def forward(self, x, pos, edge_index):
    return self.propagate(edge_index, x=x, pos=pos)

  def message(self, x_i, x_j):
    diff = x_i - x_j
    dist = (diff ** 2).sum(-1, keepdim=True).sqrt()
    msg = (dist <= self.thresholds).float()
    return msg

class ConfPredModule(torch.nn.Module):
  def __init__(self, r_min, r_max, oup_dim, **kwargs):
    super(ConfPredModule, self).__init__(**kwargs)
    self.r = r_max
    self.oup_dim = oup_dim
    self.cdf1 = CDFConv('mean', r_min, r_max, oup_dim)
    self.cdf2 = CDFConv('add', r_min, r_max, oup_dim)
    self.lin1 = torch.nn.Linear(oup_dim*2, 2)
    #cdf_good_sum = torch.zeros(oup_dim*2)
    #cdf_bad_sum = torch.zeros(oup_dim*2)
    #cdf_good_square_sum = torch.zeros(oup_dim*2)
    #cdf_bad_square_sum = torch.zeros(oup_dim*2)
    #good_count = torch.zeros(1)
    #bad_count = torch.zeros(1)
    #self.register_buffer('cdf_good_sum', cdf_good_sum)
    #self.register_buffer('cdf_bad_sum', cdf_bad_sum)
    #self.register_buffer('cdf_good_square_sum', cdf_good_square_sum)
    #self.register_buffer('cdf_bad_square_sum', cdf_bad_square_sum)
    #self.register_buffer('good_count', good_count)
    #self.register_buffer('bad_count', bad_count)
    #self.cdf_pack = []
    #self.label_pack = []

  #@property
  #def cdf_good(self):
  #  if self.good_count > 0:
  #    return self.cdf_good_sum / self.good_count
  #  else:
  #    return None

  #@property
  #def cdf_bad(self):
  #  if self.bad_count > 0:
  #    return self.cdf_bad_sum / self.bad_count
  #  else:
  #    return None

  #@property
  #def std_cdf_good(self):
  #  mean_squared = self.cdf_good_square_sum / self.good_count
  #  std = (mean_squared - (self.cdf_good ** 2)).sqrt()
  #  return std

  #@property
  #def std_cdf_bad(self):
  #  mean_squared = self.cdf_bad_square_sum / self.bad_count
  #  std = (mean_squared - (self.cdf_bad ** 2)).sqrt()
  #  return std

  #def update_cdf(self, cdf, labels):
  #  """Accumulate CDF vectors into good and bad classes.

  #  Args:
  #    cdf: [N, oup_dim*2] cdf vectors
  #    labels: [N] bool type. True indicates good.
  #  """
  #  self.cdf_good_sum += cdf[labels].sum(0)
  #  self.cdf_good_square_sum += (cdf[labels] ** 2).sum(0)
  #  self.good_count += labels.float().sum()
  #  self.cdf_bad_sum += cdf[labels == False].sum(0)
  #  self.cdf_bad_square_sum += (cdf[labels == False] ** 2).sum(0)
  #  self.bad_count += (labels == False).float().sum()
  #  self.cdf_pack.append(cdf)
  #  self.label_pack.append(labels)

  #def reset_mean_cdf(self):
  #  import ipdb; ipdb.set_trace()
  #  w0 = F.relu((self.cdf_good - self.std_cdf_good) - (self.cdf_bad + self.std_cdf_bad))
  #  w0 = w0 / (self.std_cdf_good + self.std_cdf_bad)
  #  w0 = w0.view(1, -1)
  #  #w0 = torch.ones_like(self.lin1.weight)
  #  #w0[0, self.oup_dim:] = 0.3
  #  bias_good = (self.cdf_good * w0).sum()
  #  bias_bad = (self.cdf_bad * w0).sum()
  #  b0 = (-(bias_good*0.5 + bias_bad * 0.5)).view(1)
  #  self.lin1.weight.data += w0 - self.lin1.weight.data
  #  #= torch.nn.Parameter(w0)
  #  self.lin1.bias += b0 - self.lin1.bias.data #torch.nn.Parameter(b0)

  def forward(self, x, pos, batch):
    edge_index = radius(pos, pos, self.r,
                        batch, batch, max_num_neighbors=64)
    msg1 = self.cdf1(x, pos, edge_index)
    msg2 = self.cdf2(x, pos, edge_index)
    msg = torch.cat([msg1, msg2], dim=-1)
    conf = self.lin1(msg)
    return conf, msg, edge_index

class UnpoolingModule(torch.nn.Module):
  def __init__(self, **kwargs):
    super(UnpoolingModule, self).__init__(**kwargs)

  def forward(self, msg, msg_prev, edge_idx):
    msg = msg.index_select(index=edge_idx[1], dim=0)
    if msg_prev is not None:
      msg = torch.cat([msg_prev, msg], dim=-1)
    return msg

class ConfEstModule(torch.nn.Module):
  def __init__(self, **kwargs):
    super(ConfEstModule, self).__init__(**kwargs)
    self.pool1 = PoolingModule(0.2, 0.05*1.414)
    self.pool2 = PoolingModule(0.3, 0.1*1.414)
    self.conf1 = ConfPredModule(0.05, 0.15, 10)
    self.conf2 = ConfPredModule(0.05, 0.15, 10)
    self.unpool1 = UnpoolingModule()
    self.unpool2 = UnpoolingModule()

  #def update_cdf(self, cdf_list, label_list):
  #  self.conf1.update_cdf(cdf_list[0], label_list[0])
  #  self.conf2.update_cdf(cdf_list[1], label_list[1])

  #def init_classifiers(self):
  #  self.conf1.reset_mean_cdf()
  #  self.conf2.reset_mean_cdf()

  def forward(self, x, pos, batch):
    data0_in = (x, pos, batch)
    data1_in, edge_idx10 = self.pool1(*data0_in)
    #conf1_out, cdf1_out = self.conf1(*data1_in)
    data2_in, edge_idx21 = self.pool2(*data1_in)
    x2_in, pos2_in, batch2_in = data2_in
    conf2_out, cdf2_out, edge_idx22 = self.conf2(*data2_in)
    point_weights = conf2_out.softmax(dim=-1)[:, 1:2]
    R2, trans2 = rigid_fitting(pos2_in, x2_in, edge_idx22, point_weights)
    x2_pred = (R2.matmul(pos2_in.unsqueeze(-1)) + trans2.view(-1, 3, 1)).squeeze(-1)
    x2_out = x2_pred*conf2_out[:, 0:1] + x2_in*conf2_out[:, 1:2]
    vecRT = torch.cat([R2.view(-1, 9), trans.view(-1, 3)], dim=-1)
    

    res = {}
    res['conf1'] = conf1_out
    res['conf2'] = conf2_out
    res['cdf1'] = cdf1_out
    res['cdf2'] = cdf2_out
    res['batch1'] = data1_in[2]
    res['batch2'] = data2_in[2]
    res['edge_idx1'] = edge_idx10
    res['edge_idx2'] = edge_idx21

    return res
