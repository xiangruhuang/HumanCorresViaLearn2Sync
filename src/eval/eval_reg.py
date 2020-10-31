import os, os.path as osp
import re
import glob
import progressbar
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
import torch.optim.lr_scheduler as lr_sched
import hybrid_corres as hc
from hybrid_corres.utils import helper
from hybrid_corres.config import PATH_TO_CHECKPOINT
from hybrid_corres.nn import DataParallel
from training.model import HybridModel
from training.utils import parse_args, correspondence, construct_datasets
from training.handler import MultiGPUHandler
import numpy as np
from hybrid_corres.smpl import smpl_align
import torch_geometric.transforms as T
import hybrid_corres.transforms as H
from hybrid_corres.data import Prediction

@torch.no_grad()
def test(model, loader, args, epoch, REG):
  model.train()
  handler = MultiGPUHandler(loader, args, training=False,
                            reg=True)

  CORRES = REG.replace('REG_', 'CORRES_')

  start = 0
  #import ipdb; ipdb.set_trace()
  with progressbar.ProgressBar(max_value=len(loader),
         widgets=handler.widgets) as bar:
    for i, data_list in enumerate(loader):
      if (start % 100 != 0) and ((start + 1) % 100 != 0) and ((start + 2) % 100 != 0):
        start += len(data_list)
        continue
      #print(data_list)
      centered = [T.Center()(data).to(args.device) for data in data_list]
      saved_pos = [data.pos.cpu().numpy() for data in centered]
      saved_y = [data.y.cpu().numpy() for data in centered]
      saved_pos = [torch.as_tensor(pos, dtype=torch.float).to(args.device) for pos in saved_pos]
      saved_y = [torch.as_tensor(y, dtype=torch.float).to(args.device) for y in saved_y]
      sampled = [H.GridSampling(0.01)(data).to(args.device) for data in centered]
      out_dict = model(sampled)
      corres = correspondence(out_dict['feats'], handler.template_feats)
      pred_before_reg = []
      pred_after_reg = []
      offset = 0
      x_before_reg = handler.template_points[corres, :]
      x_after_reg = out_dict['x_out0']
      
      for idx, data in enumerate(sampled):
        length = data.pos.shape[0]
        pred0 = Prediction(data.pos, x_before_reg[offset:(offset+length)])
        pred1 = Prediction(data.pos, x_after_reg[offset:(offset+length)])
        pred0 = pred0.knn_interpolate(saved_pos[idx], 3)
        pred1 = pred1.knn_interpolate(saved_pos[idx], 3)
        offset += length
        #error_smpl = np.loadtxt(CORRES.replace('.reg', '.errors').format(start+idx)).reshape(-1, 2)
        pred0.evaluate_errors(saved_y[idx][:, -3:])
        pred1.evaluate_errors(saved_y[idx][:, -3:])
        pred0.save_to_mat(CORRES.format(start+idx))
        pred1.save_to_mat(REG.format(start+idx))
        #print(errors0.mean(), errors1.mean(), error_smpl.mean(0))
        #errors = torch.stack([errors0, errors1], dim=-1)
        #np.savetxt(CORRES.format(start+idx), errors.cpu().numpy(), fmt='%.6f %.6f')
      start += len(data_list)
      #if (i % 10 == 0) or (i == len(loader)-1):
      #  handler.visualize(bar)

  torch.cuda.empty_cache()
  return {}
  

if __name__ == '__main__':
  args = parse_args()
  print(args)
  torch.cuda.manual_seed_all(816)
  model = HybridModel(args)
  if args.FEnet == 'PointNet2':
    test_datasets = []
    test_datasets.append(
      hc.datasets.SurrealFEPts(
        descriptor_dim=args.embed_dim,
        sampler=args.pc_sampler,
        split=args.test_split,
        build_graph=args.build_graph,
        cls=args.cls,
        desc=args.desc,
        transform=None,
      )
    )
    test_datasets.append(
      hc.datasets.FaustFEPts(
        descriptor_dim=args.embed_dim,
        sampler=args.pc_sampler,
        split=args.test_split,
        build_graph=args.build_graph,
        cls=args.cls,
        desc=args.desc,
        transform=None,
      )
    )
    test_datasets.append(
      hc.datasets.Shrec19FEPts(
        descriptor_dim=args.embed_dim,
        sampler=args.pc_sampler,
        split=args.test_split,
        build_graph=args.build_graph,
        cls=args.cls,
        desc=args.desc,
        transform=None,
      )
    )
  else:
    assert False, 'Model Not Implemented Yet!'

  if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)
    args.device = torch.device('cuda:0')

  model = model.to(args.device)
  lr_lbmd = lambda it: max(
    args.lr_decay ** (it * args.batch_size // args.decay_step),
    args.lr_clip / args.lr,
  )
  if args.warmstart > 0:
    ckpt = osp.join(args.ckpt_path, '%d.pt' % (args.warmstart-1))
    if osp.exists(ckpt):
      print('loading checkpoint %s' % ckpt)
      checkpoint = torch.load(ckpt, map_location=args.device)
      model_dict = model.state_dict()
      pretrained_dict = checkpoint['model_state_dict']
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      model_dict.update(pretrained_dict)
      model.load_state_dict(model_dict)

  test_loaders = [ hc.data.DataListLoader(
    test_dataset,
    batch_size=torch.cuda.device_count(),
    num_workers=6,
    shuffle=False,
    pin_memory=True,
    ) for test_dataset in test_datasets
  ]

  #CORRES = ['../data/result/SURREAL/{}.corres',
  #          '../data/result/FAUST/{}.corres',
  #          '../data/result/SHREC19/{}.corres',
  #         ]
  CORRES = ['../data/result/SURREAL/REG_{}.mat',
            '../data/result/FAUST/REG_{}.mat',
            '../data/result/SHREC19/REG_{}.mat',
           ]
  for e in range(args.warmstart, args.epochs):
    test_results = {}
    for i, test_loader in enumerate(test_loaders):
      name = test_loader.dataset.name
      print('testing {}'.format(name))
      test_results[name] = test(model, test_loader, args, e, CORRES[i])
    break
