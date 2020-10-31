import os, os.path as osp
import re
import argparse
import glob
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch_scatter import scatter_add
import progressbar
import human_corres as hc
from human_corres.data import Data
from human_corres.utils import helper
#from training.model import HybridModel
from training.utils import parse_args, correspondence, construct_datasets
from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--length', type=int, default=1)
parser.add_argument('--dataset', type=str, default=None,
  help='{SHREC19, FAUST-inter, FAUST-intra} [default: None]')
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--refined', action='store_true')
args = parser.parse_args()

if args.dataset == 'SHREC19':
  mesh_dataset = hc.datasets.Shrec19Mesh(split='test')
  MESH_CORRES = '../data/result/SHREC19/{}.mesh_corres'
  REFINED = '../data/result/SHREC19/{}.mesh_corres_icp'
  pairs, gt_indices_list = helper.read_SHREC19_pairs()
  PAIR_ERRORS = '../data/result/SHREC19/{}_{}.errors'
  PAIR_CORRES = '../data/result/SHREC19/{}_{}.npy'
elif args.dataset == 'FAUST-inter':
  mesh_dataset = hc.datasets.FaustMesh(split='test')
  MESH_CORRES = '../data/result/FAUST/{}.mesh_corres'
  REFINED = '../data/result/FAUST/{}.mesh_corres_icp'
  pairs, gt_indices_list = helper.read_FAUST_inter_pairs()
  PAIR_ERRORS = '../data/result/FAUST/{}_{}_inter.errors'
  PAIR_CORRES = '../data/result/FAUST/{}_{}_inter.npy'
elif args.dataset == 'FAUST-intra':
  mesh_dataset = hc.datasets.FaustMesh(split='test')
  MESH_CORRES = '../data/result/FAUST/{}.mesh_corres'
  REFINED = '../data/result/FAUST/{}.mesh_corres_icp'
  pairs, gt_indices_list = helper.read_FAUST_intra_pairs()
  PAIR_ERRORS = '../data/result/FAUST/{}_{}_intra.errors'
  PAIR_CORRES = '../data/result/FAUST/{}_{}_intra.npy'
elif args.dataset == 'SURREAL':
  mesh_dataset = hc.datasets.SurrealMesh(split='test')
  #MESH_CORRES = '../data/result/FAUST/{}.mesh_corres'
  REFINED = '../data/result/FAUST/{}.mesh_corres_icp'
  template_points = helper.loadSMPLModels()[0].verts
  errors_list = []
  for i in range(100):
    mesh_corres = np.loadtxt(REFINED.format(i)).astype(np.int32)[:, 1]
    mesh = mesh_dataset[i]
    errors = np.linalg.norm(template_points[mesh_corres, :] - template_points, 2, axis=-1)
    print(errors.mean())
    errors_list.append(errors)
  errors = np.concatenate(errors_list, axis=-1)
  print(errors.mean())
  print((errors<0.05).astype(np.float32).mean())
  print((errors<0.1).astype(np.float32).mean())
  assert False
  #pairs, gt_indices_list = helper.read_FAUST_intra_pairs()
  #PAIR_ERRORS = '../data/result/FAUST/{}_{}_intra.errors'
  #PAIR_CORRES = '../data/result/FAUST/{}_{}_intra.npy'

template_points = helper.loadSMPLModels()[0].verts
template_tree = NN(n_neighbors=1, n_jobs=10).fit(template_points)

if args.refined:
  print('evaluating errors of ICP Refined Correspondence')
  MESH_CORRES = REFINED
  PAIR_ERRORS = PAIR_ERRORS + '.refined'
  PAIR_CORRES = PAIR_CORRES.replace('.npy', '.refined.npy')

#errors_list = [] 
for pair, gt_indices in zip(pairs, gt_indices_list):
  print(pair)
  pair_0, pair_1 = pair
  mesh_1 = mesh_dataset[pair_1]
  mesh_points_1 = mesh_1.pos.cpu().numpy()
  corres0 = np.loadtxt(MESH_CORRES.format(pair_0)).astype(np.int32)
  corres1 = np.loadtxt(MESH_CORRES.format(pair_1)).astype(np.int32)
  pred_before_reg_0 = template_points[corres0[:, 0]]
  pred_after_reg_0 = template_points[corres0[:, 1]]
  pred_before_reg_1 = template_points[corres1[:, 0]]
  pred_after_reg_1 = template_points[corres1[:, 1]]
  tree1 = NN(n_neighbors=1, n_jobs=10).fit(pred_before_reg_1)
  _, indices_before_reg = tree1.kneighbors(pred_before_reg_0)
  #np.save(PAIR_CORRES.format(pair_0, pair_1), indices_before_reg)
  tree1 = NN(n_neighbors=1, n_jobs=10).fit(pred_after_reg_1)
  _, indices_after_reg = tree1.kneighbors(pred_after_reg_0)
  indices_before_reg = indices_before_reg[:, 0]
  indices_after_reg = indices_after_reg[:, 0]
  indices = np.stack([indices_before_reg, indices_after_reg], axis=-1)
  print('saving to {}'.format(PAIR_CORRES.format(pair_0, pair_1)))
  np.save(PAIR_CORRES.format(pair_0, pair_1), indices)
  errors_before_reg = np.linalg.norm(mesh_points_1[gt_indices] - mesh_points_1[indices_before_reg], 2, axis=-1)
  errors_after_reg = np.linalg.norm(mesh_points_1[gt_indices] - mesh_points_1[indices_after_reg], 2, axis=-1)
  errors = np.stack([errors_before_reg, errors_after_reg], axis=-1)
  print('saving to {}'.format(PAIR_ERRORS.format(pair_0, pair_1)))
  np.savetxt(PAIR_ERRORS.format(pair_0, pair_1), errors, fmt='%.6f %.6f')
  #break

#errors = np.concatenate(errors_list, axis=0)
#print(errors.mean(0))
#print((errors < 0.05).astype(np.float32).mean(0))
#print((errors < 0.1).astype(np.float32).mean(0))
