import os.path as osp

import torch
import torch.nn.functional as F
from torch_scatter import scatter, scatter_mean, scatter_add
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.nn.inits import reset
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, EdgeConv, knn
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn.conv import MessagePassing, EdgeConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from human_corres.utils import rigid_fitting, hierarchical_rigid_fitting
from human_corres.transforms import TwoHop
from human_corres.transforms import WeightedAvg
import numpy as np

class PoolingModule(torch.nn.Module):
  def __init__(self, num_clusters, **kwargs):
    super(PoolingModule, self).__init__(**kwargs)
    self.num_clusters = num_clusters

  def forward(self, x, pos, batch):
    pos6d = torch.cat([pos, x], dim=-1)
    ratio = (self.num_clusters + 2) / x.shape[0]
    idx = fps(pos6d, batch, ratio=ratio, random_start=False)
    idx = idx[:self.num_clusters]
    edge_index = knn(pos6d[idx], pos6d, 1, batch[idx], batch)
    x = scatter_mean(
          x.index_select(index=edge_index[0], dim=0),
          edge_index[1].unsqueeze(1),
          dim=0)
    pos = scatter_mean(
            pos.index_select(index=edge_index[0], dim=0),
            edge_index[1].unsqueeze(1),
            dim=0)

    return x, pos, batch[idx], edge_index

class CDFConv(MessagePassing):
  def __init__(self, aggr, r_min, r_max, num_cdfsamples, **kwargs):
    super(CDFConv, self).__init__(aggr=aggr, **kwargs)
    thresholds = np.linspace(r_min, r_max, num_cdfsamples)
    thresholds = torch.Tensor(thresholds).view(1, num_cdfsamples)
    self.register_buffer('thresholds', thresholds)

  def forward(self, x, pos, edge_index):
    return self.propagate(edge_index, x=x, pos=pos)

  def message(self, x_i, x_j, pos_i, pos_j):
    diff = x_i - x_j
    diff2 = pos_i - pos_j
    dist = (diff ** 2).sum(-1, keepdim=True).sqrt()
    dist2 = (diff2 ** 2).sum(-1, keepdim=True).sqrt()
    twist = (dist - dist2).abs()
    msg = (twist <= self.thresholds).float()
    return msg

class ConfPredModule(torch.nn.Module):
  def __init__(self, r_min, r_max, oup_dim, **kwargs):
    super(ConfPredModule, self).__init__(**kwargs)
    self.r = r_max
    self.k = 4
    self.oup_dim = oup_dim
    self.cdf1 = CDFConv('mean', r_min, r_max, oup_dim)
    self.cdf2 = CDFConv('add', r_min, r_max, oup_dim)
    self.lin1 = torch.nn.Linear(oup_dim*2, 2)

  def forward(self, x, pos, batch, edge_index):
    #row, col = radius(pos, pos, self.r,
    #                  batch, batch, max_num_neighbors=64)
    #edge_index = torch.stack([col, row], dim=0)
    #knn_index = knn(pos, pos, self.k, batch, batch)
    #edge_index = torch.cat([edge_index, knn_index], dim=-1)
    msg1 = self.cdf1(x, pos, edge_index)
    msg2 = self.cdf2(x, pos, edge_index)
    msg = torch.cat([msg1, msg2], dim=-1)
    conf = self.lin1(msg).softmax(dim=-1)
    return msg, conf

class TransfPropModule(torch.nn.Module):
  def __init__(self, **kwargs):
    super(TransfPropModule, self).__init__(**kwargs)

  def forward(self, data, cluster_weights, R, t, edge_index):
    """Propagate cluster transformations down to point transformations and compute
    transformed points.

    data: (x, pos, batch) of points. of shape [N, ?]
    cluster_weights: [M, 1] confidence of each cluster in [0, 1].
    R: rotations of clusters. [M, 3, 3]
    t: translations of clusters. [M, 3]
    edge_index: edges from points to clusters.
    """
    x, pos, batch = data
    point_idx, cluster_idx = edge_index

    R_point = R.view(-1, 9).index_select(dim=0, index=cluster_idx).view(-1, 3, 3)
    t_point = t.index_select(dim=0, index=cluster_idx)
    x_pred = (R_point.matmul(pos.unsqueeze(-1)) + t_point.unsqueeze(-1)).squeeze(-1)
    if cluster_weights is not None:
      point_weights = cluster_weights.index_select(dim=0, index=cluster_idx)
      x_out = x * point_weights + x_pred * (1.-point_weights)
    else:
      x_out = x_pred

    return R_point, t_point, x_out

class RigidPropModule(MessagePassing):
  def __init__(self, nn, rigid=True, aggr='max', **kwargs):
    super(RigidPropModule, self).__init__(aggr=aggr, **kwargs)
    self.lin1 = torch.nn.Linear(1, 1)
    self.nn = nn
    self.rigid = rigid
    self.lin1 = torch.nn.Linear(1, 1)
    self.reset_parameters()

  def reset_parameters(self):
    self.lin1.weight = torch.nn.Parameter(torch.Tensor([-1]).view(1, 1))
    self.lin1.bias = torch.nn.Parameter(torch.Tensor([0.01]).view(1))

  def forward(self, x, pos, edge_index):
    return self.propagate(x=x, pos=pos, edge_index=edge_index)

  def message(self, x_j, pos_i, pos_j):
    posi = pos_i[:, :3]
    xi = pos_i[:, 3:6]
    Ri = pos_i[:, 6:15].view(-1, 3, 3)
    ti = pos_i[:, 15:]
    posj = pos_j[:, :3]
    xj = pos_j[:, 3:6]
    msg = x_j
    if self.nn is not None:
      msg = self.nn(x_j)
    if self.rigid:
      dists = (((Ri.matmul(posj.unsqueeze(-1)) + ti.unsqueeze(-1)).squeeze(-1) - xj)
                  **2).sum(dim=-1, keepdim=True)
      weights = torch.sigmoid(self.lin1(dists))
      msg = weights * msg

    return msg

class ReweightingModule(torch.nn.Module):
  def __init__(self, num_clusters, r, init=False, **kwargs):
    super(ReweightingModule, self).__init__(**kwargs)
    self.num_clusters = num_clusters
    self.radius = r
    self.pool1 = PoolingModule(num_clusters)
    self.rigid1 = RigidPropModule(None, rigid=True)
    self.rigid2 = RigidPropModule(None, rigid=True)
    self.rigid3 = RigidPropModule(None, rigid=True)
    self.rigid4 = RigidPropModule(None, rigid=True)
    self.rigid5 = RigidPropModule(None, rigid=True)
    embedding = F.one_hot(torch.arange(start=0, end=num_clusters, dtype=torch.long),
                          num_clusters).float()
    self.register_buffer('embedding', embedding)
    #self.predictor = torch.nn.Sequential(
    #                   torch.nn.Linear(3, 2),
    #                   torch.nn.Softmax(dim=-1))
    self.predictor = torch.nn.Sequential(
                       torch.nn.Linear(5, 1),
                       torch.nn.Sigmoid())
    self.init = init
    self.reset_parameters()

  def reset_parameters(self):
    w0 = torch.ones_like(self.predictor[0].weight)
    b0 = torch.zeros_like(self.predictor[0].bias) - 17.0
    self.predictor[0].weight = torch.nn.Parameter(w0)
    self.predictor[0].bias = torch.nn.Parameter(b0)

  def knn_graph_3d(self, pos, x, batch, k, direction='source2target'):
    """Compute k-nearest neighbors in 3D Euclidean space.
    Targets indicate points and sources indicates their neighbors.
    Edges are directed either source-to-target or target-to-source.
    
    Returns:
      edge_index: [2, M] (i, j) = edge_index[:, e] indicated j is one of i's nearest neighbor.
    """
    target_idx, source_idx = knn(pos, pos, k, batch, batch)
    if direction == 'source2target':
      edge_index = torch.stack([source_idx, target_idx], dim=0)
    else:
      edge_index = torch.stack([target_idx, source_idx], dim=0)
    return edge_index

  def radius_graph_3d(self, pos, batch, r, direction='source2target'):
    target_idx, source_idx = radius(pos, pos, r, batch, batch, max_num_neighbors=32)
    if direction == 'source2target':
      edge_index = torch.stack([source_idx, target_idx], dim=0)
    else:
      edge_index = torch.stack([target_idx, source_idx], dim=0)
    return edge_index

  def estimate_transformations(self, pos0, x0, batch0):
    edge_index00_s2t = self.knn_graph_3d(pos0, x0, batch0, 10, 'source2target')
    R0, t0, point_matrices0 = rigid_fitting(pos0, x0, edge_index00_s2t,
                                reweight_x=0.01,
                                reweight_y=0.01)
    return edge_index00_s2t, R0, t0

  def transform(self, R, t, pos):
    return (R.matmul(pos.unsqueeze(-1)) + t.unsqueeze(-1)).squeeze(-1)

  def forward(self, x0, pos0, batch0, y0=None):
    edge_index00, R0, t0 = self.estimate_transformations(pos0, x0, batch0)
    x1, pos1, batch1, edge_index01 = self.pool1(x0, pos0, batch0)
    embedding = scatter_add(
                  self.embedding.index_select(dim=0, index=edge_index01[1]),
                  edge_index01[0].unsqueeze(-1),
                  dim=0, dim_size=x0.shape[0],
                  )
    radius_index00 = self.radius_graph_3d(pos0, batch0, self.radius)
    vec0 = torch.cat([pos0, x0, R0.view(-1, 9), t0.view(-1, 3)], dim=-1)
    msg1 = self.rigid1(x=embedding, pos=vec0, edge_index=radius_index00)
    msg2 = self.rigid2(x=msg1, pos=vec0, edge_index=radius_index00)
    msg3 = self.rigid3(x=msg2, pos=vec0, edge_index=radius_index00)
    msg4 = self.rigid4(x=msg2, pos=vec0, edge_index=radius_index00)
    msg5 = self.rigid5(x=msg2, pos=vec0, edge_index=radius_index00)
    msg = torch.cat([msg1.sum(-1, keepdim=True),
                     msg2.sum(-1, keepdim=True),
                     msg3.sum(-1, keepdim=True),
                     msg4.sum(-1, keepdim=True),
                     msg5.sum(-1, keepdim=True),
                    ], dim=-1)
    if self.init:
      return msg
    else:
      conf0 = self.predictor(msg)[:, 0:1]
      
      R_out0 = WeightedAvg()(R0.view(-1, 9), conf0, edge_index00, 'source2target').view(-1, 3, 3)
      t_out0 = WeightedAvg()(t0, conf0, edge_index00, 'source2target')

      x_out0 = self.transform(R_out0, t_out0, pos0)
      return x_out0, R_out0, t_out0

class RegularizationModule(torch.nn.Module):
  def __init__(self, init=False, **kwargs):
    super(RegularizationModule, self).__init__(**kwargs)
    self.init = init
    self.reweight1 = ReweightingModule(400, 0.1, init=self.init)

  def forward(self, x, pos, batch, y):
    x0, pos0, batch0 = (x, pos, batch)
    if self.init:
      msg = self.reweight1(x0, pos0, batch0, y)
      res = {}
      res['msg'] = msg
    else:
      x_r1, R_r1, t_r1 = self.reweight1(x0, pos0, batch0, y)
      x_out = x_r1
      res = {}
      res['x_out0'] = x_out

    for i, data in enumerate([(x0, pos0, batch0),
                              ]):
      res['batch{}'.format(i)] = data[2]
      res['pos{}'.format(i)] = data[1]
      res['x{}'.format(i)] = data[0]
    return res

class Regularization2Module(torch.nn.Module):
  def __init__(self, **kwargs):
    super(Regularization2Module, self).__init__(**kwargs)
    self.reweight1 = ReweightingModule(400, 0.15)
    self.reweight2 = ReweightingModule(600, 0.12)

  def forward(self, x, pos, batch, y):
    x0, pos0, batch0 = (x, pos, batch)
    x_r1, R_r1, t_r1 = self.reweight1(x0, pos0, batch0)
    x_r2, R_r2, t_r2 = self.reweight2(x_r1, pos0, batch0)
    x_out = x_r2
    res = {}
    res['x_out0'] = x_out

    for i, data in enumerate([(x0, pos0, batch0),
                              ]):
      res['batch{}'.format(i)] = data[2]
      res['pos{}'.format(i)] = data[1]
      res['x{}'.format(i)] = data[0]
    return res

class Regularization3Module(torch.nn.Module):
  def __init__(self, **kwargs):
    super(Regularization3Module, self).__init__(**kwargs)
    self.reweight1 = ReweightingModule(400, 0.15)
    self.reweight2 = ReweightingModule(600, 0.12)
    self.reweight3 = ReweightingModule(800, 0.10)

  def forward(self, x, pos, batch, y):
    x0, pos0, batch0 = (x, pos, batch)
    x_r1, R_r1, t_r1 = self.reweight1(x0, pos0, batch0)
    x_r2, R_r2, t_r2 = self.reweight2(x_r1, pos0, batch0)
    x_r3, R_r3, t_r3 = self.reweight3(x_r2, pos0, batch0)
    x_out = x_r3
    res = {}
    res['x_out0'] = x_out

    for i, data in enumerate([(x0, pos0, batch0),
                              ]):
      res['batch{}'.format(i)] = data[2]
      res['pos{}'.format(i)] = data[1]
      res['x{}'.format(i)] = data[0]
    return res
