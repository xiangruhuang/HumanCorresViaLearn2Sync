import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np
from human_corres.config import PATH_TO_SHREC19
from human_corres.utils import helper
from torch_geometric.data import Data
import scipy.io as sio
import torch_geometric.transforms as T
import human_corres.transforms as H
import open3d as o3d

class Shrec19Mesh(Dataset):
  """SHREC19-Human 3D points for Feature Extraction (FE).

  Output: dictionary with keys {points3d, correspondence}
  Data Format:
    points3d: [num_points, 3] real numbers.
    correspondence: [num_points] integers in range [6890].
  """
  def __init__(self, split='train'):
    super(Shrec19Mesh).__init__()
    self.name = 'Shrec19FEMesh'
    self.num_views = 100
    self.IDlist = np.arange(1, 45)
    self.split = split
    if self.split == 'train':
      raise RuntimeError("This dataset is Test Only")
    elif self.split == 'test':
      self.IDlist = self.IDlist
    elif self.split == 'val':
      self.IDlist = self.IDlist[:2]
    self.PLY = '{}/mesh/{{}}.ply'.format(PATH_TO_SHREC19)
    self.CORRES = '{}/mesh/{{}}.corres'.format(PATH_TO_SHREC19)

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index
    mesh = o3d.io.read_triangle_mesh(self.PLY.format(mesh_id))
    points3d = np.array(mesh.vertices)
    points3d = torch.as_tensor(points3d, dtype=torch.float)
    corres = np.loadtxt(self.CORRES.format(mesh_id)).astype(np.int32)
    y = torch.Tensor(corres).long()
    faces = torch.as_tensor(mesh.triangles, dtype=torch.long)
    data = Data(pos=points3d, y=y, faces=faces)
    return data

  def __len__(self):
    return len(self.IDlist)
