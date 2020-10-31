import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_scatter import scatter
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate


class IterativeModule(torch.nn.Module):
  """Extract Point Transformations from point features.
  
  """
  def __init__(self, sigma):
    super(IterativeModule, self).__init__()
    self.sigma = sigma
    self.sigma2 = sigma*sigma

  def forward(self, source_points, target_points, batch, edge_index, edge_weights=None):
    """Compute Point-wise Rigid Transformation.

    Args:
      source_feats (Tensor): point-wise feature vectors
        with shape [num_edges, feat_dim].
      target_feats (Tensor): precomputed feature vectors of template points
        with shape [6890, feat_dim].
      data (torch_geometric.data.Data): the input point cloud.
    """
    N = source_points.shape[0]
    src_neighbor_locations = torch.index_select(source_points, dim=0, index=edge_index[1])
    tgt_neighbor_locations = torch.index_select(target_points, dim=0, index=edge_index[1])
    src_centers = scatter(src_neighbor_locations, edge_index[0], dim=0, dim_size=N, reduce='mean')
    tgt_centers = scatter(tgt_neighbor_locations, edge_index[0], dim=0, dim_size=N, reduce='mean')
    src_neighbor_diffs = src_neighbor_locations - torch.index_select(src_centers, 0, edge_index[0])
    tgt_neighbor_diffs = tgt_neighbor_locations - torch.index_select(tgt_centers, 0, edge_index[0])
    if edge_weights is not None:
      src_neighbor_diffs = src_neighbor_diffs*edge_weights
    edge_matrices = torch.matmul(
                      src_neighbor_diffs.unsqueeze(-1), 
                      tgt_neighbor_diffs.unsqueeze(-2)
                    )
    point_matrices = scatter(edge_matrices, edge_index[0], dim=0, dim_size=N, reduce='sum')
    U, S, V = point_matrices.svd()
    with torch.no_grad():
      R0 = torch.matmul(U, V.transpose(1, 2))
      D = torch.diag_embed(torch.stack([torch.ones(N).to(R0.device), torch.ones(N).to(R0.device), R0.det()], axis=-1))
    R = torch.matmul(torch.matmul(U, D), V.transpose(1, 2))
    trans = tgt_centers - torch.matmul(R, src_centers.unsqueeze(-1)).squeeze(-1)
    src_neighbor_rotations = torch.index_select(R, dim=0, index=edge_index[1])
    src_neighbor_translations = torch.index_select(trans, dim=0, index=edge_index[1])
    src_neighbor_transformed = (torch.matmul(src_neighbor_rotations, src_neighbor_locations.view(-1, 3, 1)) + src_neighbor_translations.view(-1, 3, 1)).view(-1, 3)
    dsts = src_neighbor_transformed - tgt_neighbor_locations
    dsts = (dsts*dsts).sum(-1, keepdim=True)
    edge_weights = self.sigma2 / (dsts + self.sigma2)

    return R, trans, edge_weights

class TransformationModule(torch.nn.Module):
  """Extract Point Transformations from point features.
  
  """
  def __init__(self, feat_dim):
    super(TransformationModule, self).__init__()
    self.feat_dim = feat_dim

  def forward(self, source_feats, target_feats, target_points, data):
    """Compute Point-wise Rigid Transformation.

    Args:
      source_feats (Tensor): point-wise feature vectors
        with shape [num_edges, feat_dim].
      target_feats (Tensor): precomputed feature vectors of template points
        with shape [6890, feat_dim].
      data (torch_geometric.data.Data): the input point cloud.
    """
    N = source_feats.shape[0]
    dp = torch.matmul(source_feats, target_feats.transpose(0, 1))
    source_squares = (source_feats*source_feats).sum(dim=-1, keepdim=True)
    target_squares = (target_feats*target_feats).sum(dim=-1, keepdim=True)
    dists = source_squares - 2*dp + target_squares.view(1, -1)
    scores, indices = torch.topk(-dists, k=6)
    scores = scores.softmax(dim=-1)
    selected_points = torch.index_select(target_points, 0, indices.view(-1)).view(-1, 6, 3)
    predicted_points = (selected_points * scores.unsqueeze(-1)).sum(dim=-2)
    src_neighbor_locations = torch.index_select(data.pos, dim=0, index=data.edge_index[1, :])
    tgt_neighbor_locations = torch.index_select(predicted_points, dim=0, index=data.edge_index[1, :])
    src_centers = scatter(src_neighbor_locations, data.edge_index[0], dim=0, dim_size=N, reduce='mean')
    tgt_centers = scatter(tgt_neighbor_locations, data.edge_index[0], dim=0, dim_size=N, reduce='mean')
    src_neighbor_diffs = src_neighbor_locations - torch.index_select(src_centers, 0, data.edge_index[0])
    tgt_neighbor_diffs = tgt_neighbor_locations - torch.index_select(tgt_centers, 0, data.edge_index[0])
    edge_matrices = torch.matmul(src_neighbor_diffs.unsqueeze(-1), tgt_neighbor_diffs.unsqueeze(-2))
    point_matrices = scatter(edge_matrices, data.edge_index[0], dim=0, dim_size=N, reduce='sum')
    U, S, V = point_matrices.svd()
    with torch.no_grad():
      R0 = torch.matmul(U, V.transpose(1, 2))
      D = torch.diag_embed(torch.stack([torch.ones(N).to(R0.device), torch.ones(N).to(R0.device), R0.det()], axis=-1))
    R = torch.matmul(torch.matmul(U, D), V.transpose(1, 2))
    trans = tgt_centers - torch.matmul(R, src_centers.unsqueeze(-1)).squeeze(-1)
    src_neighbor_locations = torch.index_select(data.pos, dim=0, index=data.edge_index[1, :])
    src_neighbor_rotations = torch.index_select(R, dim=0, index=data.edge_index[1, :])
    src_neighbor_translations = torch.index_select(trans, dim=0, index=data.edge_index[1, :])
    src_neighbor_transformed = (torch.matmul(src_neighbor_rotations, src_neighbor_locations.view(-1, 3, 1)) + src_neighbor_translations.view(-1, 3, 1)).view(-1, 3)
    dsts = src_neighbor_transformed - tgt_neighbor_locations
    dsts = (dsts*dsts).sum(-1, keepdim=True)
    dst = scatter(dsts, data.edge_index[0], dim=0, dim_size=N, reduce='mean')
    #dst = ((torch.matmul(R, data.pos.view(-1, 3, 1)) + trans.view(-1, 3, 1)).squeeze(-1) - predicted_points)
    return R, trans, dst
