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
import human_corres as hc
import human_corres.transforms as H

num_test = 5000
TrainTransform = T.Compose([
  T.Center(),
  T.RandomRotate(30, axis=0),
  T.RandomRotate(30, axis=1),
  T.RandomRotate(30, axis=2),
  H.GridSampling(0.01)
])
TestTransform = T.Compose([
  T.Center(),
  H.GridSampling(0.01)
])

class SurrealFEPts(Dataset):
  """Surreal 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, descriptor_dim, split='train', desc='Laplacian_n',
               transform='default', cls=False):
    super(SurrealFEPts).__init__()
    self.name = 'SurrealFEPts'
    if (split == 'train') or (split == 'val'):
      self.num_views = 20
    else:
      self.num_views = 100
    self.IDlist = np.stack([np.arange(100000*self.num_views),
                            np.arange(115000*self.num_views,
                                      215000*self.num_views)],
                           axis=0)
    self.split = split
    if self.split == 'train':
      self.IDlist = self.IDlist[:, :-(num_test*self.num_views)].reshape(-1)
      self.file_path = '{}/scans/{{0:06d}}/{{1:03d}}.mat'.format(hc.PATH_TO_SURREAL)
    elif self.split == 'test':
      self.IDlist = self.IDlist[:, -(num_test*self.num_views):].reshape(-1)
      self.file_path = '{}/scans/{{0:06d}}/{{1:03d}}.mat'.format(hc.PATH_TO_SURREAL_TEST)
    elif self.split == 'val':
      self.IDlist = self.IDlist[:, :(5*self.num_views)].reshape(-1)
      self.file_path = '{}/scans/{{0:06d}}/{{1:03d}}.mat'.format(hc.PATH_TO_SURREAL)
      #ll = [20,24,25,26,28,35,36,37,39,52,53,106,128,152,160,178,187,191]
      #self.IDlist = np.array([self.IDlist[l] for l in ll])
    self.result_path = '{}/result/SURREAL/'.format(hc.PATH_TO_DATA)
    self.template_feats = helper.loadSMPLDescriptors(desc)[:, :descriptor_dim]
    self.template_points = helper.loadSMPLModels()[0].verts
    self.cls = cls
    if transform == 'default':
      if self.split == 'train':
        self.transform = TrainTransform
      else:
        self.transform = TestTransform
    else:
      self.transform = transform

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index // self.num_views
    view_id = index % self.num_views
    filename = self.file_path.format(mesh_id, view_id)
    mat = sio.loadmat(filename, variable_names=['points3d', 'correspondences',
                                                'rotations'])
    points3d = mat['points3d']
    corres = mat['correspondences'].reshape(-1).astype(np.int64)
    corres = torch.from_numpy(corres)
    num_points = points3d.shape[0]
    points3d = torch.Tensor(points3d)
    R = mat['rotations'].reshape(3, 3)
    ori_pos = torch.as_tensor(R.T.dot(points3d.numpy().T).T, dtype=torch.float)
    if not self.cls:
      gt_feats = self.template_feats[corres, :]
      gt_points = self.template_points[corres, :]
      y = np.concatenate([gt_feats, gt_points], axis=-1)
      y = torch.Tensor(y)
    else:
      y = torch.as_tensor(corres, dtype=torch.long)

    data = Data(pos=points3d, y=y, ori_pos=ori_pos)
    data = self.transform(data) if self.transform is not None else data

    return data

  def __len__(self):
    return len(self.IDlist)

if __name__ == '__main__':
  surreal_pts = SurrealFEPts(descriptor_dim=100, split='test')
  data_loader = torch_geometric.data.DataLoader(
    surreal_pts,
    batch_size=10,
    pin_memory=True,
    num_workers=6,
    shuffle=True,
    )
  for data in data_loader:
    print(data.ori_pos[:10], data.pos[:10])
    print(data.pos.shape, data.ori_pos.shape)
    break

