import os.path as osp
import shutil

import torch
from torch.utils.data import Dataset
import numpy as np
from human_corres.config import PATH_TO_DATA
from human_corres.utils import helper
from human_corres.data import Data
import scipy.io as sio
import open3d as o3d
import glob

class ToscaMesh(Dataset):
  """TOSCA 3D Triangle Mesh for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, split='train', obj_class='cat'):
    super(ToscaMesh).__init__()
    self.name = 'ToscaMesh'
    self.num_views = 100
    self.split = split
    self.obj_class = obj_class
    self.PLY = '{}/TOSCA/mesh/{}{{}}.ply'.format(PATH_TO_DATA, self.obj_class)
    self.IDlist = np.arange(len(glob.glob(self.PLY.format('*'))))
    #self.IDlist = sorted(self.IDlist, key = lambda s: int(s.strip().split('/')[-1].split('.')[0].replace(self.obj_class, '')))
    print(self.IDlist)
    if self.split == 'train':
      self.IDlist = self.IDlist[:-3]
    elif self.split == 'test':
      self.IDlist = self.IDlist[-3:]
    elif self.split == 'val':
      self.IDlist = self.IDlist[:5]

  def __getitem__(self, i):
    mesh_id = self.IDlist[i]
    mesh = o3d.io.read_triangle_mesh(self.PLY.format(mesh_id))
    points3d = torch.as_tensor(np.stack([x, y, z], -1), dtype=torch.float)
    corres = np.arange(points3d.shape[0]).astype(np.int32)
    mesh = o3d.io.read_triangle_mesh(self.PLY.format(mesh_id))
    points3d = np.array(mesh.vertices)
    corres = np.arange(points3d.shape[0])
    points3d = torch.as_tensor(points3d, dtype=torch.float)
    faces = torch.as_tensor(mesh.triangles, dtype=torch.long)
    y = torch.as_tensor(corres, dtype=torch.long)
    data = Data(pos=points3d, faces=faces, y=y)
    return data

  def __len__(self):
    return len(self.IDlist)

