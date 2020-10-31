import argparse
import scipy.io as sio
from human_corres.utils import helper
from human_corres.smpl import smpl_align_mesh
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
import torch
import human_corres as hc

parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--length', type=int, default=1)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--step', type=int, default=1)
args = parser.parse_args()
num_views = 100

if args.dataset == 'SHREC19':
  mesh_dataset = hc.datasets.Shrec19Mesh(split='test')
  RESULT = '../data/result/SHREC19/{}.mat'
  MESH_CORRES = '../data/result/SHREC19/{}.mesh_corres'
  REFINED = '../data/result/SHREC19/{}.mesh_corres_icp'
elif args.dataset == 'FAUST':
  mesh_dataset = hc.datasets.FaustMesh(split='test')
  RESULT = '../data/result/FAUST/{}.mat'
  MESH_CORRES = '../data/result/FAUST/{}.mesh_corres'
  REFINED = '../data/result/FAUST/{}.mesh_corres_icp'
elif args.dataset == 'SURREAL':
  mesh_dataset = hc.datasets.SurrealMesh(split='test')
  RESULT = '../data/result/SURREAL/{}.mat'
  MESH_CORRES = '../data/result/SURREAL/{}.mesh_corres'
  REFINED = '../data/result/SURREAL/{}.mesh_corres_icp'
elif args.dataset == 'FAUST-Test':
  mesh_dataset = hc.datasets.FaustTestMesh(split='test')
  RESULT = '../data/result/FAUST-Test/{}.mat'
  MESH_CORRES = '../data/result/FAUST-Test/{}.mesh_corres'
  REFINED = '../data/result/FAUST-Test/{}.mesh_corres_icp'
  num_views = 200

template_points = helper.loadSMPLModels()[0].verts
template_tree = NN(n_neighbors=1, n_jobs=10).fit(template_points)
def get_result(mesh_id, mesh):
  ori_pos = []
  pred_before_reg = []
  pred_after_reg = []
  mesh_points = mesh.pos.cpu().numpy()
  for view_id in range(num_views):
    idx = mesh_id * num_views + view_id
    res_dict = sio.loadmat(RESULT.format(idx))
    ori_pos.append(res_dict['ori_pos'])
    pred_before_reg.append(res_dict['pred_before_reg'])
    pred_after_reg.append(res_dict['pred_after_reg'])
  ori_pos = np.concatenate(ori_pos, axis=0)
  tree = NN(n_neighbors=3, n_jobs=10).fit(ori_pos)
  dists, indices = tree.kneighbors(mesh_points)
  print(dists.mean())
  weights = (0.01**2)/(0.01**2+dists**2)
  weights = weights / weights.sum(-1)[:, np.newaxis]
  pred_before_reg = np.concatenate(pred_before_reg, axis=0)
  pred_after_reg = np.concatenate(pred_after_reg, axis=0)
  mesh_pred_before_reg = (pred_before_reg[indices]*weights[:, :, np.newaxis]).sum(-2)
  mesh_pred_after_reg = (pred_after_reg[indices]*weights[:, :, np.newaxis]).sum(-2)
  _, corres_before_reg = template_tree.kneighbors(mesh_pred_before_reg)
  _, corres_after_reg = template_tree.kneighbors(mesh_pred_after_reg)
  return corres_before_reg[:, 0], corres_after_reg[:, 0] 

for mesh_id in range(args.offset, args.offset+args.length):
  mesh = mesh_dataset[mesh_id]
  print('working on ', mesh_id)
  vertices = mesh.pos.cpu().numpy()
  faces = mesh.faces.cpu().numpy()
  corres0, corres1 = get_result(mesh_id, mesh)
  if mesh.y is None:
    gt_corres = None
  else:
    gt_corres = mesh.y.cpu().numpy().astype(np.int32)
    errors0 = np.linalg.norm(template_points[gt_corres, :] - template_points[corres0, :], 2, axis=-1)
    errors1 = np.linalg.norm(template_points[gt_corres, :] - template_points[corres1, :], 2, axis=-1)
    errors = np.stack([errors0, errors1], axis=-1)
    print('errors before icp: ', errors.mean(0))
  corres = np.stack([corres0, corres1], axis=-1)
  np.savetxt(MESH_CORRES.format(mesh_id), corres, fmt='%d %d')
  new_corres0 = smpl_align_mesh(vertices, faces, corres0, max_iter=50)
  new_corres1 = smpl_align_mesh(vertices, faces, corres1, max_iter=50)
  new_corres = np.stack([new_corres0, new_corres1], axis=-1)
  if gt_corres is not None:
    errors0 = np.linalg.norm(template_points[gt_corres, :] - template_points[new_corres0, :], 2, axis=-1)
    errors1 = np.linalg.norm(template_points[gt_corres, :] - template_points[new_corres1, :], 2, axis=-1)
    errors = np.stack([errors0, errors1], axis=-1)
    print('errors after icp: ', errors.mean(0))
  np.savetxt(REFINED.format(mesh_id), new_corres, fmt='%d %d')
