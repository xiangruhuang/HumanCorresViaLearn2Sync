import os, os.path as osp
import re
import glob
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
import torch.optim.lr_scheduler as lr_sched
import progressbar
import human_corres as hc
from human_corres.utils import helper
from human_corres.config import PATH_TO_CHECKPOINT
from human_corres.modules import HybridModel
from .single_gpu_handler import Handler
from .utils import parse_args, correspondence
from .utils import construct_datasets, construct_animal_datasets

def train(model, loader, optimizer, lr_scheduler, args, epoch):
  model.train()
  handler = Handler(loader, args, lr_scheduler,
                    training=True, reg=args.transf_reg)

  with progressbar.ProgressBar(max_value=len(loader),
                               widgets=handler.widgets) as bar:
    for i, data in enumerate(loader):
      data = data.to(args.device)
      optimizer.zero_grad()
      out_dict = model(data)
      loss = handler.parse(out_dict, data)
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      if ((i + 1) % 100 == 0) or (i == len(loader) - 1):
        handler.visualize(bar)
      if (i + 1) % 100 == 0:
        torch.save({
          'epoch': epoch,
          'afteriter': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          }, osp.join(args.save_path, '%d.pt' % (epoch))
          )
  torch.cuda.empty_cache()

@torch.no_grad()
def test(model, loader, args, epoch):
  model.train() # this is better for batchnorm
  handler = Handler(loader, args, None, training=False, reg=args.transf_reg)
 
  if args.dump_result:
    MAT = '{}/{{}}.mat'.format(loader.dataset.result_path)
    os.system('mkdir -p {}'.format(loader.dataset.result_path))
    start = 0

  with progressbar.ProgressBar(max_value=len(loader), 
                               widgets=handler.widgets) as bar:
    for i, data in enumerate(loader):
      data = data.to(args.device)
      out_dict = model(data)
      if args.dump_result:
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
      if ((i + 1) % 100 == 0) or (i == len(loader) - 1):
        handler.visualize(bar)
      if (i + 1) % 100 == 0:
        torch.save({
          'epoch': epoch,
          'afteriter': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          }, osp.join(args.save_path, '%d.pt' % (epoch))
          )
  torch.cuda.empty_cache()

if __name__ == '__main__':
  args = parse_args()
  print(args)
  torch.cuda.manual_seed_all(816)
  model = HybridModel(args)
  if args.animals:
    train_dataset, test_datasets = construct_animal_datasets(args)
  else:
    train_dataset, test_datasets = construct_datasets(args)

  model = model.to(args.device)
  lr_lbmd = lambda it: max(
    args.lr_decay ** (it * args.batch_size // args.decay_step),
    args.lr_clip / args.lr,
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
      pretrained_dict = converted_dict
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      model_dict.update(pretrained_dict)
      model.load_state_dict(model_dict)
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

  train_loader = hc.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=True,
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

  for e in range(args.warmstart, args.epochs):
    if not args.testing:
      train(model, train_loader, optimizer, lr_scheduler, args, e)
    for test_loader in test_loaders:
      name = test_loader.dataset.name
      print('testing {}'.format(name))
      test(model, test_loader, args, e)
    if not args.testing:
      save_dict = {}
      save_dict['epoch'] = e
      save_dict['model_state_dict'] = model.state_dict()
      save_dict['optimizer_state_dict'] = optimizer.state_dict()
      save_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
      torch.save(
        save_dict,
        osp.join(args.ckpt_path, '%d.pt' % (e))
      )
    else:
      break
