import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np
from hybrid_corres.data import ImgData, ImgBatch
from hybrid_corres.config import PATH_TO_FAUST
from hybrid_corres.utils import helper
import scipy.io as sio

class FaustFEDepthImgs(Dataset):
  """FAUST Depth Images for feature extraction (FE).

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
  def __init__(self, descriptor_dim, split='train'):
    super(FaustFEDepthImgs).__init__()
    self.name = 'FaustFEDepthImgs'
    self.IDlist = np.arange(10000)
    self.split = split
    if self.split == 'train':
      raise RuntimeError("This dataset is Test Only")
      self.IDlist = self.IDlist
    elif self.split == 'test':
      self.IDlist = self.IDlist
    elif self.split == 'val':
      self.IDlist = self.IDlist[:40]
    self.file_path = '{}/scans/{{0:03d}}_{{1:03d}}.mat'.format(PATH_TO_FAUST)
    self.template_feats = helper.loadSMPLDescriptors()[:, :descriptor_dim]
    self.template_points = helper.loadSMPLModels()[0].verts

  def __getitem__(self, i):
    index = self.IDlist[i]
    mesh_id = index // 100
    view_id = index % 100
    filename = self.file_path.format(mesh_id, view_id)
    mat = sio.loadmat(filename,
      variable_names=['valid_pixel_indices', 'depth', 'correspondence',
                      'width', 'height', 'points3d'])
    points3d = mat['points3d']
    W = int(mat['width'])
    H = 256 #int(mat['height'])
    valid_idx = mat['valid_pixel_indices'].astype(np.int64)
    depth = mat['depth']
    corres = mat['correspondence'].reshape(-1)
    image = helper.depth2image(depth, points3d, valid_idx, H, W)
    image = np.transpose(image, (2, 0, 1))

    gt_feats = self.template_feats[corres, :]
    gt_points = self.template_points[corres, :]
    y = np.concatenate([gt_feats, gt_points], axis=-1)
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
