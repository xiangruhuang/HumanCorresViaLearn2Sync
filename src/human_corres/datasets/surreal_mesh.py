import torch
import os, sys
import os.path as osp
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from human_corres.utils import helper
from torch_geometric.data import Data
import torch_geometric
import torch_geometric.transforms as T
import human_corres.transforms as H
import human_corres as hc

num_views = 100
IDlist = np.stack([np.arange(100000),
                   np.arange(115000, 215000)],
                  axis=0)

num_test = 50

class SurrealMesh(Dataset):
  """Surreal 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, split='train'):
    super(SurrealMesh).__init__()
    self.name = 'SurrealMesh'
    self.num_views = 100
    self.split = split
    self.IDlist = IDlist
    if self.split == 'train':
      assert False, 'SurrealMesh is test only!'
    elif self.split == 'test':
      self.IDlist = self.IDlist[:, -num_test:].reshape(-1)
    elif self.split == 'val':
      self.IDlist = self.IDlist[:, :5].reshape(-1)
    self.models = helper.loadSMPLModels()
    param_file = hc.PATH_TO_SURREAL_PARAMS
    self.surreal_params = sio.loadmat(param_file)['params']

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index
    params = self.surreal_params[mesh_id]
    params[11:14] = 0.0
    gender = int(params[0])
    model = self.models[gender]
    params = np.concatenate([np.zeros(3), params[1:]], axis=-1)
    model.update_params(params)
    points3d = torch.as_tensor(model.verts, dtype=torch.float)
    faces = torch.as_tensor(model.faces.astype(np.int32), dtype=torch.long)
    corres = np.arange(6890)

    y = torch.as_tensor(corres, dtype=torch.long)

    data = Data(pos=points3d, y=y, faces=faces)

    return data

  def __len__(self):
    return len(self.IDlist)
