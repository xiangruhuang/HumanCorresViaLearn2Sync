import argparse
import scipy.io as sio
from hybrid_corres.utils import helper
from hybrid_corres.smpl import smpl_align
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
#import torch_geometric.transforms as T
#from torch_geometric.data import Data
from hybrid_corres.data import Prediction
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--length', type=int, default=1)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--step', type=int, default=1)

args = parser.parse_args()

if args.dataset == 'SURREAL':
  num_views = 20
  IDlist = np.stack([np.arange(100000*num_views),
                     np.arange(115000*num_views, 215000*num_views)],
                    axis=0)
  num_test = 50
  IDlist = IDlist[:, -(num_test*num_views):].reshape(-1)
  MAT='../data/surreal/scans/{0:06d}/{1:03d}.mat'
elif args.dataset == 'FAUST':
  IDlist = np.arange(10000)
  num_views = 100
  MAT='../data/faust/scans/{0:03d}_{1:03d}.mat'
elif args.dataset == 'SHREC19':
  IDlist = np.arange(100, 4500)
  num_views = 100
  MAT='../data/SHREC19/scans/{0:03d}_{1:03d}.mat'
CORRES='../data/result/{}/{{}}.corres'.format(args.dataset)
PRED='../data/result/{}/SMPL_{{}}.mat'.format(args.dataset)
ERRORS='../data/result/{}/{{}}.errors'.format(args.dataset)

model = helper.loadSMPLModels()[0]
template_points = torch.as_tensor(model.verts, dtype=torch.float)
tree = NN(n_neighbors=1).fit(template_points)
for i in range(args.offset, args.offset+args.length, args.step):
  packed = np.loadtxt(CORRES.format(i)).reshape(-1, 7)
  sampled_points = torch.as_tensor(packed[:, :3], dtype=torch.float)
  gt_points = torch.as_tensor(packed[:, 3:6], dtype=torch.float)
  sampled_corres = torch.as_tensor(packed[:, -1].astype(np.int32), dtype=torch.long)
  pred = Prediction(pos=sampled_points, x=template_points[sampled_corres])

  idx = IDlist[i]
  mesh_id = idx // num_views
  view_id = idx % num_views
  mat = sio.loadmat(MAT.format(mesh_id, view_id))
  print(MAT.format(mesh_id, view_id))
  points = mat['points3d']
  points = points - points.mean(0)[np.newaxis, :]
  points = torch.as_tensor(points, dtype=torch.float)
  pred = pred.knn_interpolate(points, k=3)
  _, corres = tree.kneighbors(pred.x.numpy())
  corres = torch.as_tensor(corres[:, 0], dtype=torch.long)
  if args.dataset == 'FAUST' or args.dataset == 'SHREC19':
    key = 'correspondence'
  else:
    key = 'correspondences'
  gt_corres = mat[key].reshape(-1).astype(np.int32)
  gt_corres = torch.as_tensor(gt_corres, dtype=torch.long)
  
  #import ipdb; ipdb.set_trace()
  new_corres = smpl_align(sampled_points.numpy(), sampled_corres.numpy(), max_iter=15)
  new_corres = torch.as_tensor(new_corres, dtype=torch.long)
  pred_after = Prediction(pos=sampled_points, x=template_points[new_corres,:])
  pred_after = pred_after.knn_interpolate(points, k=3)
  
  errors_before = (template_points[gt_corres, :] - template_points[corres, :]).norm(p=2, dim=-1)
  errors_after = (template_points[gt_corres, :] - pred_after.x).norm(p=2, dim=-1)
  #errors_before = np.linalg.norm(template_points[gt_corres, :] - template_points[corres, :], 2, axis=-1)
  #errors_after = np.linalg.norm(template_points[gt_corres, :] - template_points[new_corres, :], 2, axis=-1)
  #pred_after = Prediction(pos=points, x=te)
  pred_after.evaluate_errors(template_points[gt_corres, :])
  pred_after.save_to_mat(PRED.format(i))
  #errors = np.stack([errors_before, errors_after], axis=-1)
  #np.savetxt(ERRORS.format(i), errors, fmt='%.6f %.6f')
  print(errors_before.mean(), errors_after.mean())
  #break
