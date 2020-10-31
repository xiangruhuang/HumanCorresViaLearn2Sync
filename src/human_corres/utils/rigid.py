import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_scatter import scatter_add
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils import add_self_loops, remove_self_loops

def hierarchical_rigid_fitting(
      source_points,
      target_points,
      point2cluster,
      num_clusters,
      point_weights=None,
      reweight_x=-1,
      reweight_y=-1):
  """Compute Patch-wise Rigid Transformation. Patches are centered around a subset of points.

  Args:
    source_points: [N, 3] Tensor.
    target_points: [N, 3] Tensor.
    point2cluster: [2, M] LongTensor, directed edges from points to clusters.
    point_weights: [N, 1] weights in (0, 1).
    reweight_x: if True, reweight edges according to distances in source domain.
    reweight_y: if True, reweight edges according to distances in target domain.
  """
  N = source_points.shape[0]
  M = num_clusters
  point_idx, cluster_idx = point2cluster

  # compute patch-wise centers (weighted)
  if point_weights is None:
    point_weights = torch.ones(N, 1).to(source_points.device).float()
  edge_weights = point_weights.index_select(0, index=point_idx)

  src_neighbor_locations = source_points.index_select(0, index=point_idx)
  tgt_neighbor_locations = target_points.index_select(0, index=point_idx)
  src_locations = source_points.index_select(0, index=cluster_idx)
  if reweight_x > 0:
    src_dists = ((src_locations - src_neighbor_locations)**2).sum(dim=-1, keepdim=True)
    reweights_x = (reweight_x**2)/(src_dists+reweight_x**2)
    edge_weights = edge_weights * reweights_x
  tgt_locations = target_points.index_select(0, index=cluster_idx)
  if reweight_y > 0:
    tgt_dists = ((tgt_locations - tgt_neighbor_locations)**2).sum(dim=-1, keepdim=True)
    reweights_y = (reweight_y**2)/(tgt_dists+reweight_y**2)
    edge_weights = edge_weights * reweights_y

  src_centers_sum = scatter_add(
                      src_neighbor_locations*edge_weights,
                      cluster_idx, dim=0, dim_size=M)
  src_weight_sums = scatter_add(edge_weights, cluster_idx, dim=0, dim_size=M)
  src_centers = src_centers_sum / src_weight_sums
  tgt_centers_sum = scatter_add(
                      tgt_neighbor_locations*edge_weights,
                      cluster_idx, dim=0, dim_size=M)
  tgt_weight_sums = scatter_add(edge_weights, cluster_idx, dim=0, dim_size=M)
  tgt_centers = tgt_centers_sum / tgt_weight_sums

  # compute patch-wise accumulated rank-one matrices
  src_neighbor_diffs = src_neighbor_locations - src_locations
  tgt_neighbor_diffs = tgt_neighbor_locations - tgt_locations

  edge_matrices = torch.matmul(
                    tgt_neighbor_diffs.unsqueeze(-1),
                    (src_neighbor_diffs*edge_weights).unsqueeze(-2)
                  )
  point_matrices = scatter_add(
                     edge_matrices, cluster_idx,
                     dim=0, dim_size=M)
  try:
    U, S, V = point_matrices.svd()
  except Exception as e:
    print(e)
    import ipdb; ipdb.set_trace()
    print(e)
  #try:
  #  with torch.no_grad():
  #    #invalid_mask = (S.min(dim=-1)[0] < 1e-7).unsqueeze(-1).unsqueeze(-1)
  #    R0 = U.matmul(V.transpose(1, 2))
  #    D = torch.diag_embed(torch.stack([torch.ones(M).to(R0.device), torch.ones(M).to(R0.device), R0.det()], axis=-1))
  #except Exception as e:
  #  import ipdb; ipdb.set_trace()
  #  print(e)
  #R = U.matmul(D).matmul(V.transpose(1, 2))
  R = U.matmul(V.transpose(1, 2))
  trans = tgt_centers - torch.matmul(R, src_centers.unsqueeze(-1)).squeeze(-1)

  return R, trans, point_matrices

def rigid_fitting(source_points, target_points, edge_index,
                  point_weights=None, direction='point2cluster',
                  reweight_x=-1,
                  reweight_y=-1):
  """Compute Patch-wise Rigid Transformation. Each patch is centers around one point.

  Args:
    source_points: [N, 3] Tensor.
    target_points: [N, 3] Tensor.
    edge_index: [2, M] LongTensor, directed edges from edge_index[1] to edge_index[0].
    point_weights: [N, 1] weights in (0, 1).
  """
  N = source_points.shape[0]
  edge_index, _ = remove_self_loops(edge_index)
  edge_index, _ = add_self_loops(edge_index, num_nodes=N)
  if direction=='point2cluster':
    point_idx, cluster_idx = edge_index
  else:
    cluster_idx, point_idx = edge_index

  # compute patch-wise centers (weighted)
  if point_weights is None:
    point_weights = torch.ones(N, 1).to(source_points.device)
  edge_weights = point_weights.index_select(dim=0, index=point_idx)

  src_neighbor_locations = source_points.index_select(0, index=point_idx)
  tgt_neighbor_locations = target_points.index_select(0, index=point_idx)
  if reweight_x > 0:
    src_locations = source_points.index_select(0, index=cluster_idx)
    src_dists = ((src_locations - src_neighbor_locations)**2).sum(dim=-1, keepdim=True)
    reweights_x = (reweight_x**2)/(src_dists+reweight_x**2)
    edge_weights = edge_weights * reweights_x
  if reweight_y > 0:
    tgt_locations = target_points.index_select(0, index=cluster_idx)
    tgt_dists = ((tgt_locations - tgt_neighbor_locations)**2).sum(dim=-1, keepdim=True)
    reweights_y = (reweight_y**2)/(tgt_dists+reweight_y**2)
    edge_weights = edge_weights * reweights_y

  src_centers_sum = scatter_add(
                      src_neighbor_locations*edge_weights,
                      cluster_idx, dim=0, dim_size=N)
  src_weight_sums = scatter_add(edge_weights, cluster_idx, dim=0, dim_size=N)
  src_centers = src_centers_sum / src_weight_sums
  tgt_centers_sum = scatter_add(
                      tgt_neighbor_locations*edge_weights,
                      cluster_idx, dim=0, dim_size=N)
  tgt_weight_sums = scatter_add(edge_weights, cluster_idx, dim=0, dim_size=N)
  tgt_centers = tgt_centers_sum / tgt_weight_sums

  # compute patch-wise accumulated rank-one matrices
  src_neighbor_diffs = src_neighbor_locations - src_centers.index_select(0, cluster_idx)
  tgt_neighbor_diffs = tgt_neighbor_locations - tgt_centers.index_select(0, cluster_idx)

  edge_matrices = torch.matmul(
                    tgt_neighbor_diffs.unsqueeze(-1),
                    (src_neighbor_diffs*edge_weights).unsqueeze(-2)
                  )
  point_matrices = scatter_add(
                     edge_matrices, cluster_idx,
                     dim=0, dim_size=N)
  try:
    U, S, V = point_matrices.svd()
  except Exception as e:
    print(e)
    import ipdb; ipdb.set_trace()
    print(e)
  #with torch.no_grad():
  #  invalid_mask = (S.min(dim=-1)[0] < 1e-7).unsqueeze(-1).unsqueeze(-1)
  #  R0 = torch.matmul(U, V.transpose(1, 2))
  #  D = torch.diag_embed(torch.stack([torch.ones(N).to(R0.device), torch.ones(N).to(R0.device), R0.det()], axis=-1))
  R = torch.matmul(U, V.transpose(1, 2))
  trans = tgt_centers - torch.matmul(R, src_centers.unsqueeze(-1)).squeeze(-1)
  #src_neighbor_rotations = torch.index_select(R, dim=0, index=edge_index[1])
  #src_neighbor_translations = torch.index_select(trans, dim=0, index=edge_index[1])
  #src_neighbor_transformed = (torch.matmul(src_neighbor_rotations, src_neighbor_locations.view(-1, 3, 1)) + src_neighbor_translations.view(-1, 3, 1)).view(-1, 3)
  #dsts = src_neighbor_transformed - tgt_neighbor_locations
  #dsts = (dsts*dsts).sum(-1, keepdim=True)
  #edge_weights = self.sigma2 / (dsts + self.sigma2)

  #return None, None, None
  return R, trans, point_matrices

