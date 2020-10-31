import os, os.path as osp
import re
import glob
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate
from torch_scatter import scatter_add
import progressbar
import hybrid_corres as hc
from hybrid_corres.data import Data
from hybrid_corres.utils import helper
from training.model import HybridModel
from training.utils import parse_args, correspondence, construct_datasets
from sklearn.neighbors import NearestNeighbors as NN
import numpy as np
import scipy.io as sio

def eval_pairs(mesh_corres_list, mesh_points_list, gt_indices_list, pairs):
  errors_list = []
  for pair, gt_indices in zip(pairs, gt_indices_list):
    i, j = pair
    mesh_corres_i = mesh_corres_list[i]
    mesh_corres_j = mesh_corres_list[j]
    mesh_points_j = mesh_points_list[j]
    tree = NN(n_neighbors=1, n_jobs=10).fit(mesh_corres_j)
    _, indices = tree.kneighbors(mesh_corres_i)
    indices = indices[:, 0]
    errors = np.linalg.norm(mesh_points_j[indices, :] - mesh_points_j[gt_indices, :], 2, axis=-1)
    errors_list.append(errors)
    print(pair)
  errors_vec = np.concatenate(errors_list, axis=0)
  print(errors_vec.mean(), np.mean((errors_vec < 0.1).astype(np.float32)))

@torch.no_grad()
def test(model, scan_loader, mesh_loader, num_views, args):
  model.eval()

  handler = Handler(scan_loader, mesh_loader, num_views, args)
  with progressbar.ProgressBar(max_value=len(mesh_loader), widgets=handler.widgets) as bar:
    for mesh_id, scans, mesh in zip(range(len(mesh_loader)), scan_loader, mesh_loader):
      handler.parse_mesh(mesh)
      y, pos, batch = scans.y, scans.pos, scans.batch
      for i in range(num_views):
        batch_mask = (batch == i)
        yi = y[batch_mask]
        posi = pos[batch_mask]
        #batchi = batch[batch_mask]
        data = Batch.from_data_list([Data(y=yi, pos=posi)])
        data = data.to(args.device)
        out_dict = model(data)
        handler.parse(out_dict, data, i)
      handler.visualize(bar)
  return handler.mesh_corres_list, handler.mesh_points_list, handler.gt_mesh_corres_list


def eval_mesh2mesh(mesh_dataset, num_views, args, RESULT, pairs, PAIR_ERRORS):
  def get_result(mesh_id):
    ori_pos = []
    pred_before_reg = []
    pred_after_reg = []
    mesh = mesh_dataset[mesh_id]
    mesh_points = mesh.pos.cpu().numpy()
    for view_id in range(num_views):
      idx = mesh_id * num_views + view_id
      res_dict = sio.loadmat(RESULT.format(idx))
      ori_pos.append(res_dict['ori_pos'])
      pred_before_reg.append(res_dict['pred_before_reg'])
      pred_after_reg.append(res_dict['pred_after_reg'])
    ori_pos = np.concatenate(ori_pos, axis=0)
    tree = NN(n_neighbors=5, n_jobs=10).fit(ori_pos)
    dists, indices = tree.kneighbors(mesh_points)
    weights = (0.01**2)/(0.01**2+dists**2)
    weights = weights / weights.sum(-1)[:, np.newaxis]
    pred_before_reg = np.concatenate(pred_before_reg, axis=0)
    pred_after_reg = np.concatenate(pred_after_reg, axis=0)
    mesh_pred_before_reg = (pred_before_reg[indices]*weights[:, :, np.newaxis]).sum(-2)
    mesh_pred_after_reg = (pred_after_reg[indices]*weights[:, :, np.newaxis]).sum(-2)
    return mesh_pred_before_reg, mesh_pred_after_reg

  result_dict = {}
  pairs, gt_indices_list = pairs
  for pair, gt_indices in zip(pairs, gt_indices_list):
    pair_0, pair_1 = pair
    mesh_0 = mesh_dataset[pair_0]
    mesh_1 = mesh_dataset[pair_1]
    if result_dict.get(pair_0, None) is not None:
      pred_before_reg_0, pred_after_reg_0 = result_dict[pair_0]
    else:
      pred_before_reg_0, pred_after_reg_0 = get_result(pair_0)
      result_dict[pair_0] = (pred_before_reg_0, pred_after_reg_0)
    if result_dict.get(pair_1, None) is not None:
      pred_before_reg_1, pred_after_reg_1 = result_dict[pair_1]
    else:
      pred_before_reg_1, pred_after_reg_1 = get_result(pair_1)
      result_dict[pair_1] = (pred_before_reg_1, pred_after_reg_1)
    tree1 = NN(n_neighbors=1, n_jobs=10).fit(pred_before_reg_1)
    _, indices_before_reg = tree1.kneighbors(pred_before_reg_0)
    indices_before_reg = indices_before_reg[:, 0]
    tree1 = NN(n_neighbors=1, n_jobs=10).fit(pred_after_reg_1)
    _, indices_after_reg = tree1.kneighbors(pred_after_reg_0)
    indices_after_reg = indices_after_reg[:, 0]
    mesh_points_1 = mesh_1.pos.cpu().numpy()
    errors_before_reg = np.linalg.norm(mesh_points_1[gt_indices] - mesh_points_1[indices_before_reg], 2, axis=-1)
    errors_after_reg = np.linalg.norm(mesh_points_1[gt_indices] - mesh_points_1[indices_after_reg], 2, axis=-1)
    errors = np.stack([errors_before_reg, errors_after_reg], axis=-1)
    np.savetxt(PAIR_ERRORS.format(pair_0, pair_1), errors, fmt='%.6f %.6f')

if __name__ == '__main__':
  args = parse_args()
  print(args)
  torch.cuda.manual_seed_all(816)
  #model = HybridModel(args)
  if args.FEnet == 'PointNet2':
    train_dataset, _, mesh_datasets = construct_datasets(args, mesh=True)
  else:
    assert False, 'PointNet2 Only!'

  #model = model.to(args.device)
  #if args.warmstart > 0:
  #  ckpt = osp.join(args.ckpt_path, '%d.pt' % (args.warmstart-1))
  #  if osp.exists(ckpt):
  #    print('loading checkpoint %s' % ckpt)
  #    checkpoint = torch.load(ckpt, map_location=args.device)
  #    model_dict = model.state_dict()
  #    pretrained_dict = checkpoint['model_state_dict']
  #    converted_dict = {}
  #    for key, val in pretrained_dict.items():
  #      if key.startswith('module.'):
  #        converted_dict[re.sub('^module.', '', key)] = val
  #    pretrained_dict = converted_dict
  #    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  #    model_dict.update(pretrained_dict)
  #    model.load_state_dict(model_dict)

  #train_loader = hc.data.DataLoader(
  #  train_dataset,
  #  batch_size=args.batch_size,
  #  num_workers=6,
  #  shuffle=True,
  #  pin_memory=True,
  #)
  #mesh_loaders = [ hc.data.DataLoader(
  #  mesh_dataset,
  #  batch_size=1,
  #  num_workers=6,
  #  shuffle=False,
  #  pin_memory=True,
  #  ) for mesh_dataset in mesh_datasets
  #]

  res_dicts = [
               #'../data/result/SHREC19/{}.mat',
               '../data/result/FAUST/{}.mat',
               '../data/result/FAUST/{}.mat',
               #'../data/result/SURREAL/{}.mat',
              ]

  errors_dicts = [
                  '../data/result/FAUST/{}_{}_intra.errors',
                  '../data/result/FAUST/{}_{}_inter.errors',
                 ]

  pairs_list = [
                #helper.read_SHREC19_pairs(),
                helper.read_FAUST_intra_pairs(),
                helper.read_FAUST_inter_pairs()]

  faust_mesh_dataset = mesh_datasets[1]
  mesh_datasets = [faust_mesh_dataset, faust_mesh_dataset]
  #mesh_datasets = mesh_datasets[1:2]

  for mesh_dataset, RESULT, pairs, ERRORS in zip(mesh_datasets, res_dicts, pairs_list, errors_dicts):
    name = mesh_dataset.name
    print('testing {}'.format(name))
    num_views = mesh_dataset.num_views
    eval_mesh2mesh(mesh_dataset, num_views, args, RESULT, pairs, ERRORS)
    #eval_pairs(mesh_corres_list, mesh_points_list, gt_indices_list, pairs)
    #break
