import torch
import os, sys
import os.path as osp
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from human_corres.utils import helper
from human_corres.data import Data
import torch_geometric
import torch_geometric.transforms as T
from human_corres.data import ImgData, ImgBatch
from human_corres.config import PATH_TO_SURREAL_PARAMS, PATH_TO_SURREAL
import human_corres.transforms as H

num_views = 20
IDlist = np.stack([np.arange(100000*num_views),
                   np.arange(115000*num_views, 215000*num_views)],
                  axis=0)
num_test = 50

class SurrealFEDepthImgs(Dataset):
  """Surreal Depth Images for feature extraction (FE).

  D = descriptor dimension.
  H = height (default 240).
  W = width (default 320).
  Output: dictionary with keys {valid_indices, points3d, image, gt_feats,
                                gt_points, correspondence}
  Data Format:
    valid_indices: [num_points, 2] integers for pixel coordinates.
    points3d: [num_points, 3] 3D scan points.
    image: [H, W, 5] a depth image with channels = [mask, depth, x, y, z]
    gt_feats: [num_points, D] ground truth features (on template).
    gt_points: [num_points, 3] ground truth corresponding points (on template).
    correspondence: [num_points] integers in range [6890] (on template).
  """
  def __init__(self, num_points, descriptor_dim, split='train'):
    super(SurrealFEDepthImgs).__init__()
    self.name = 'SurrealFEDepthImgs'
    self.split = split
    self.num_sample_points = num_points
    if self.split == 'train':
      self.IDlist = IDlist[:, :-(num_test*num_views)].reshape(-1)
    elif self.split == 'test':
      self.IDlist = IDlist[:, -(num_test*num_views):].reshape(-1)
    elif self.split == 'val':
      self.IDlist = IDlist[:, :num_views].reshape(-1)
    #self.file_path = '{}/scans/%d_%d.mat'.format(PATH_TO_SURREAL)
    self.file_path = '{}/scans/{{0:06d}}/{{1:03d}}.mat'.format(PATH_TO_SURREAL)
    self.template_feats = helper.loadSMPLDescriptors()[:, :descriptor_dim]
    self.template_points = helper.loadSMPLModels()[0].verts

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index // num_views
    view_id = index % num_views
    filename = self.file_path.format(mesh_id, view_id)
    mat = sio.loadmat(filename,
      variable_names=['valid_pixel_indices', 'depths', 'correspondences',
                      'width', 'height', 'points3d'])
    points3d = mat['points3d']
    W = int(mat['width'])
    H = 256 #int(mat['height'])
    valid_idx = mat['valid_pixel_indices'].astype(np.int64)
    depth = mat['depths']
    corres = mat['correspondences'].reshape(-1)
    image = helper.depth2image(depth, points3d, valid_idx, H, W)
    image = np.transpose(image, (2, 0, 1))

    gt_feats = self.template_feats[corres, :]
    gt_points = self.template_points[corres, :]
    y = np.concatenate([gt_feats, gt_points], axis=-1)
    #points3d = torch.Tensor(points3d)
    valid_idx = torch.from_numpy(valid_idx)
    y = torch.Tensor(y)
    image = torch.Tensor(image)
    mask = np.zeros((H, W)).astype(np.int64)
    mask[(valid_idx[:, 0], valid_idx[:, 1])] = 1
    mask = torch.Tensor(mask)
    data = ImgData(img=image, y=y, indices=valid_idx, mask=mask)

    return data

  def __len__(self):
    return len(self.IDlist)
