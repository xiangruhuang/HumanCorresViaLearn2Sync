import os, os.path as osp
import re
import glob
import torch
import numpy as np
import progressbar
import argparse
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch.optim.lr_scheduler as lr_sched
from torch_scatter import scatter_mean
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn
from torch_geometric.nn import knn_interpolate
from torch_geometric.utils.metric import precision, recall, true_positive, true_negative, false_positive, false_negative
import hybrid_corres as hc
from hybrid_corres.utils import helper
from hybrid_corres.config import PATH_TO_CHECKPOINT, PATH_TO_RESULT
from training.model import HybridModel
from hybrid_corres.utils import visualization as vis
from hybrid_corres.utils import rigid_fitting
from .utils import parse_args, correspondence, construct_datasets
from .handler import Handler

def train(model, loader, optimizer, lr_scheduler, args, epoch):
  model.train()

  handler = Handler(loader, args, lr_scheduler, training=True, reg=True)

  with progressbar.ProgressBar(max_value=len(loader), widgets=handler.widgets) as bar:
    for i, data in enumerate(loader):
      data = data.to(args.device)
      optimizer.zero_grad()
      out_dict = model(data)
      loss = handler.parse(out_dict, data)
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      if ((i+1) % 10 == 0) or (i == len(loader)-1):
        handler.visualize(bar)
        torch.save({
          'epoch': epoch,
          'afteriter': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          }, osp.join(args.save_path, '{}.pt'.format(epoch))
        )
  torch.cuda.empty_cache()
  return {}

@torch.no_grad()
def test(model, loader, args, epoch, dump_result=False):
  model.train()
  handler = Handler(loader, args, None, training=False, reg=True)

  if dump_result:
    MAT = '{}/{{}}.mat'.format(loader.dataset.result_path)
    os.system('mkdir -p {}'.format(loader.dataset.result_path))
    start = 0
  with progressbar.ProgressBar(max_value=len(loader), widgets=handler.widgets) as bar:
    for i, data in enumerate(loader):
      data = data.to(args.device)
      out_dict = model(data)
      if dump_result:
        loss, res_dict = handler.parse(out_dict, data, require_corres=True)
        for idx, data in enumerate(data_list):
          sio.savemat(
            MAT.format(start),
            {'ori_pos': res_dict['ori_pos'][idx],
             'pred_before_reg': res_dict['pred_before_reg'][idx],
             'pred_after_reg': res_dict['pred_after_reg'][idx]}
          )
          start += 1
      else:
        loss = handler.parse(out_dict, data)

      if ((i + 1) % 100 == 0) or (i == len(loader)-1):
        handler.visualize(bar)

  torch.cuda.empty_cache()
  return {}

if __name__ == '__main__':
  args = parse_args()
  print(args)
  torch.cuda.manual_seed_all(816)
  model = HybridModel(args)
  if args.FEnet == 'PointNet2':
    train_dataset, test_datasets = construct_datasets(args)
  else:
    assert False, 'Model Not Implemented Yet!'

  model = model.to(args.device)
  lr_lbmd = lambda it: max(
    args.lr_decay ** (it * args.batch_size // args.decay_step),
    args.lr_clip / args.lr,
  )
  conf_params = [param for name, param in model.named_parameters()
                  if name.startswith('reg')]
  optimizer = torch.optim.Adam(conf_params, lr=args.lr)
  lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=-1)
  if args.warmstart > 0:
    ckpt = osp.join(args.load_path, '%d.pt' % (args.warmstart-1))
    if osp.exists(ckpt):
      print('loading checkpoint %s' % ckpt)
      checkpoint = torch.load(ckpt, map_location=args.device)
      model_dict = model.state_dict()
      pretrained_dict = checkpoint['model_state_dict']
      converted_dict = {}
      for key, val in pretrained_dict.items():
        if key.startswith('module.'):
          converted_dict[re.sub('^module.', '', key)] = val
        else:
          converted_dict[key] = val
      pretrained_dict = converted_dict
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      model_dict.update(pretrained_dict)
      model.load_state_dict(model_dict)
      #if checkpoint.get('optimizer_state_dict', None) is not None:
      #  try:
      #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      #  except Exception as e:
      #    print(e)
      #if checkpoint.get('lr_scheduler_state_dict', None) is not None:
      #  try:
      #    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
      #  except Exception as e:
      #    print(e)

  train_loader = hc.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=False,
    pin_memory=True,
  )
  test_loaders = [ hc.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=False,
    pin_memory=True,
    ) for test_dataset in test_datasets
  ]

  #initialize_cdf(model, train_loader, args)
  for e in range(args.warmstart, args.warmstart + args.epochs):
    if not args.testing:
      train(model, train_loader, optimizer, lr_scheduler, args, e)
    #test_results = {}
    #for test_loader in test_loaders:
    #  name = test_loader.dataset.name
    #  print('testing {}'.format(name))
    #  test(model, test_loader, args, e)
    if not args.testing:
      save_dict = {}
      save_dict['epoch'] = e
      save_dict['model_state_dict'] = model.state_dict()
      save_dict['optimizer_state_dict'] = optimizer.state_dict()
      save_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
      #save_dict['train_loss'] = train_result['avg_loss']
      #for name, test_result in test_results.items():
      #  save_dict['test_l2dist_{}'.format(name)] = test_result
      torch.save(
        save_dict,
        osp.join(args.save_path, '{}.pt'.format('latest'))
      )
    else:
      break
