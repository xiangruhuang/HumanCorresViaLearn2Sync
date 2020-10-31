import os.path as osp
import shutil

import torch
from torch.utils.data import Dataset
import numpy as np
from human_corres.config import PATH_TO_DATA
from human_corres.utils import helper
from torch_geometric.data import Data
import scipy.io as sio
import open3d as o3d

class FaustMesh(Dataset):
  """Faust 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, split='train'):
    super(FaustMesh).__init__()
    self.name = 'FaustMesh'
    self.num_views = 100
    self.IDlist = np.arange(100)
    self.split = split
    if self.split == 'train':
      raise RuntimeError("This dataset is Test Only")
    elif self.split == 'test':
      self.IDlist = self.IDlist
    elif self.split == 'val':
      self.IDlist = self.IDlist[:5]

    self.PLY = '{}/MPI-FAUST/training/registrations/tr_reg_{{0:03d}}.ply'.format(PATH_TO_DATA)
    self.CORRES = '{}/MPI-FAUST/training/registrations/tr_reg_{{0:03d}}.corres'.format(PATH_TO_DATA)

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index
    mesh = o3d.io.read_triangle_mesh(self.PLY.format(mesh_id))
    corres = np.loadtxt(self.CORRES.format(mesh_id)).astype(np.int32)
    points3d = np.array(mesh.vertices)
    points3d = torch.as_tensor(points3d, dtype=torch.float)
    faces = torch.as_tensor(mesh.triangles, dtype=torch.long)
    y = torch.as_tensor(corres, dtype=torch.long)
    data = Data(pos=points3d, faces=faces, y=y)
    return data

  def __len__(self):
    return len(self.IDlist)

class FaustTestMesh(Dataset):
  """Faust 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, split='train'):
    super(FaustTestMesh).__init__()
    self.name = 'FaustTestMesh'
    self.num_views = 200
    self.IDlist = np.arange(200)
    self.split = split
    if self.split == 'train':
      raise RuntimeError("This dataset is Test Only")
    elif self.split == 'test':
      self.IDlist = self.IDlist
    elif self.split == 'val':
      self.IDlist = self.IDlist[:5]

    self.PLY = '{}/MPI-FAUST/test/scans/test_scan_{{0:03d}}.ply'.format(PATH_TO_DATA)
    #self.CORRES = '{}/MPI-FAUST/test/scans/test_scan_{{0:03d}}.corres'.format(PATH_TO_DATA)

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index
    mesh = o3d.io.read_triangle_mesh(self.PLY.format(mesh_id))
    #corres = np.loadtxt(self.CORRES.format(mesh_id)).astype(np.int32)
    points3d = np.array(mesh.vertices)
    points3d = torch.as_tensor(points3d, dtype=torch.float)
    faces = torch.as_tensor(mesh.triangles, dtype=torch.long)
    #y = torch.as_tensor(corres, dtype=torch.long)
    data = Data(pos=points3d, faces=faces)
    return data

  def __len__(self):
    return len(self.IDlist)
