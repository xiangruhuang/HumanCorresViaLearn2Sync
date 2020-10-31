import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.io import read_ply
from torch.utils.data import Dataset
import numpy as np
from hybrid_corres.data import ImgData, ImgBatch
from hybrid_corres.config import PATH_TO_DATA
from hybrid_corres.utils import helper
from torch_geometric.data import Data
import scipy.io as sio
import torch_geometric.transforms as T
import hybrid_corres.transforms as H

DefaultTransform = T.Compose([
  T.Center(),
  T.RandomRotate(30, axis=0),
  T.RandomRotate(30, axis=1),
  T.RandomRotate(30, axis=2),
])

class FaustFEPts5k(Dataset):
  """Faust 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, descriptor_dim, sampler=None, split='train',
               transform=DefaultTransform, build_graph=False, cls=False):
    super(FaustFEPts5k).__init__()
    self.name = 'FaustFEPts5k'
    self.IDlist = np.arange(10000)
    self.split = split
    if self.split == 'train':
      raise RuntimeError("This dataset is Test Only")
    elif self.split == 'test':
      self.IDlist = self.IDlist
    elif self.split == 'val':
      self.IDlist = self.IDlist[:40]
    self.file_path = '{}/faust/scans/{{0:03d}}_{{0:03d}}.mat'.format(PATH_TO_DATA)
    self.template_feats = helper.loadSMPLDescriptors()[:, :descriptor_dim]
    self.template_points = helper.loadSMPLModels()[0].verts
    self.pre_transform = None #T.NormalizeScale()
    self.cls = cls
    if build_graph:
      self.transform = T.Compose([transform, T.KNNGraph(k=6), T.ToDense(5000)])
    else:
      self.transform = T.Compose([transform, T.ToDense(5000)])

  def voxel_sample(self, pos, corres):
    fake_data = Data(
      pos=torch.Tensor(pos),
      y=corres,
    )
    data = self.voxel_sampler(fake_data)
    return data.pos, data.y

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index // 100
    view_id = index % 100
    filename = self.file_path.format(mesh_id, view_id)
    mat = sio.loadmat(filename, variable_names=['points3d', 'correspondence'])
    points3d = mat['points3d']
    num_points = points3d.shape[0]
    corres = mat['correspondence'].reshape(-1).astype(np.int64)
    corres = mat['correspondences'].reshape(-1).astype(np.int64)
    sample_idx = np.random.randint(0, num_points, size=5000)
    points3d = points3d[sample_idx]
    corres = corres[sample_idx]
    corres = torch.from_numpy(corres)
    points3d = torch.Tensor(points3d)
    if self.cls:
      y = torch.as_tensor(corres, dtype=torch.long)
    else:
      gt_feats = self.template_feats[corres, :]
      gt_points = self.template_points[corres, :]
      y = np.concatenate([gt_feats, gt_points], axis=-1)
      y = torch.Tensor(y)
    data = Data(pos=points3d, y=y)
    data = self.transform(data) if self.transform is not None else data
    return data

  def __len__(self):
    return len(self.IDlist)
