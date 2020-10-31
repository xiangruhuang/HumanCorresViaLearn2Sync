import os.path as osp

import torch
from torch.utils.data import Dataset
import numpy as np
from human_corres.config import PATH_TO_DATA
from human_corres.utils import helper
from human_corres.data import Data
import scipy.io as sio
import open3d as o3d
import glob

import human_corres as hc
import scipy.io as sio
import torch_geometric.transforms as T
import human_corres.transforms as H

TrainTransform = T.Compose([
  T.Center(),
  T.RandomRotate(30, axis=0),
  T.RandomRotate(30, axis=1),
  T.RandomRotate(30, axis=2),
  H.GridSampling(0.01),
])

TestTransform = T.Compose([
  T.Center(),
  H.GridSampling(0.01),
])

class ToscaFEPts(Dataset):
  """Tosca 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, descriptor_dim, split='train', desc='Laplacian_n',
               transform='default', cls=False):
    super(ToscaFEPts).__init__()
    self.name = 'ToscaFEPts'
    self.num_views = 100
    self.result_path = '{}/result/TOSCA/'.format(hc.PATH_TO_DATA)
    self.IDlist = np.arange(1200)
    self.split = split
    if self.split == 'train':
      raise RuntimeError("This dataset is Test Only")
    elif self.split == 'test':
      self.IDlist = self.IDlist
    elif self.split == 'val':
      self.IDlist = self.IDlist[:40]
    self.file_path = '{}/scans/{{0:03d}}_{{1:03d}}.mat'.format(
                        hc.PATH_TO_TOSCA_SCANS)
    self.template_feats = helper.loadAnimalDescriptors(desc)[:, :descriptor_dim]
    self.template_points = helper.loadAnimalModels()[0].verts
    self.cls = cls
    if transform == 'default':
      if self.split == 'train':
        self.transform = TrainTransform
      else:
        self.transform = TestTransform
    else:
      self.transform = transform

  def voxel_sample(self, pos, corres):
    fake_data = Data(
      pos=torch.Tensor(pos),
      y=corres,
    )
    data = self.voxel_sampler(fake_data)
    return data.pos, data.y

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index // self.num_views
    view_id = index % self.num_views
    filename = self.file_path.format(mesh_id, view_id)
    mat = sio.loadmat(filename, variable_names=['points3d', 'correspondence',
                                                'rotation'])
    points3d = mat['points3d']
    corres = mat['correspondence'].reshape(-1).astype(np.int64)
    corres = torch.from_numpy(corres)
    num_points = points3d.shape[0]
    points3d = torch.Tensor(points3d)
    R = mat['rotation'].reshape(3, 3)
    ori_pos = torch.as_tensor(R.T.dot(points3d.numpy().T).T, dtype=torch.float)
    if self.cls:
      y = torch.as_tensor(corres, dtype=torch.long)
    else:
      gt_feats = self.template_feats[corres, :]
      gt_points = self.template_points[corres, :]
      y = np.concatenate([gt_feats, gt_points], axis=-1)
      y = torch.Tensor(y)
    data = Data(pos=points3d, y=y, ori_pos=ori_pos)
    data = self.transform(data) if self.transform is not None else data
    return data

  def __len__(self):
    return len(self.IDlist)
  
