import torch
import os, sys
import os.path as osp
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
#abs_path = os.path.abspath(__file__)
#abs_path = os.path.dirname(abs_path)
#abs_path = os.path.dirname(abs_path)
#abs_path = os.path.dirname(abs_path)
#project_path = os.path.dirname(abs_path)
#sys.path.append(project_path)
from human_corres.utils import helper
from torch_geometric.data import Data
import torch_geometric
#from torch_geometric.transforms import GridSampling
import torch_geometric.transforms as T
from human_corres.data import ImgData, ImgBatch
from human_corres.config import PATH_TO_SURREAL
import human_corres.transforms as H

num_views = 20
IDlist = np.stack([np.arange(100000*num_views),
                   np.arange(115000*num_views, 215000*num_views)],
                  axis=0)
num_test = 5000
DefaultTransform = T.Compose([
  T.Center(),
  T.RandomRotate(30, axis=0),
  T.RandomRotate(30, axis=1),
  T.RandomRotate(30, axis=2),
])

class SurrealFEPts5k(Dataset):
  """Surreal 3D points for Feature Extraction (FE).
  Samples a fixed number of points.

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, descriptor_dim, sampler=None, split='train',
               transform=DefaultTransform, cls=False, build_graph=False):
    super(SurrealFEPts5k).__init__()
    self.name = 'SurrealFEPts5k'
    self.split = split
    if self.split == 'train':
      self.IDlist = IDlist[:, :-(num_test*num_views)].reshape(-1)
    elif self.split == 'test':
      self.IDlist = IDlist[:, -(num_test*num_views):].reshape(-1)
    elif self.split == 'val':
      self.IDlist = IDlist[:, :num_views].reshape(-1)
    self.file_path = '{}/scans/{{0:06d}}/{{1:03d}}.mat'.format(PATH_TO_SURREAL)
    self.template_feats = helper.loadSMPLDescriptors()[:, :descriptor_dim]
    self.template_points = helper.loadSMPLModels()[0].verts
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
    mesh_id = index // num_views
    view_id = index % num_views
    filename = self.file_path.format(mesh_id, view_id)
    mat = sio.loadmat(filename, variable_names=['points3d', 'correspondences'])
    points3d = mat['points3d']
    num_points = points3d.shape[0]
    corres = mat['correspondences'].reshape(-1).astype(np.int64)
    sample_idx = np.random.randint(0, num_points, size=5000)
    points3d = points3d[sample_idx]
    corres = corres[sample_idx]
    corres = torch.from_numpy(corres)
    points3d = torch.Tensor(points3d)
    if not self.cls:
      gt_feats = self.template_feats[corres, :]
      gt_points = self.template_points[corres, :]
      y = np.concatenate([gt_feats, gt_points], axis=-1)
      y = torch.Tensor(y)
    else:
      y = torch.as_tensor(corres, dtype=torch.long)

    data = Data(pos=points3d, y=y)
    data = self.transform(data) if self.transform is not None else data

    return data

  def __len__(self):
    return len(self.IDlist)
