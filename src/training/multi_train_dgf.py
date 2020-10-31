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
from .utils import parse_args, correspondence, construct_datasets
from .handler import MultiGPUHandler

def train(model, loader, optimizer, lr_scheduler, args, epoch):
  model.train()

  handler = MultiGPUHandler(loader, args, training=True, reg=False)

  with progressbar.ProgressBar(max_value=len(loader),
         widgets=handler.widgets) as bar:
    for i, data_list in enumerate(loader):
      optimizer.zero_grad()
      out_dict = model(data_list)
      loss = handler.parse(out_dict, data_list)
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      if ((i + 1) % 10 == 0) or (i == len(loader)-1):
        handler.visualize(bar)
        torch.save({
          'epoch': epoch,
          'afteriter': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          }, osp.join(args.ckpt_path, '%d.pt' % (epoch))
        )
  torch.cuda.empty_cache()
  return {}

@torch.no_grad()
def test(model, loader, args, epoch):
  model.train()
  handler = MultiGPUHandler(loader, args, training=False,
                            reg=False)

  with progressbar.ProgressBar(max_value=len(loader),
         widgets=handler.widgets) as bar:
    for i, data_list in enumerate(loader):
      out_dict = model(data_list)
      loss = handler.parse(out_dict, data_list)

      if (i % 10 == 0) or (i == len(loader)-1):
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
    train_dataset = hc.datasets.DGFSurrealFEPts(
      descriptor_dim=args.embed_dim,
      split=args.train_split,
      desc=args.desc,
      cls=args.cls,
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
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=-1)
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
      #model.load_state_dict(checkpoint['model_state_dict'])
      #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      #lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

  train_loader = hc.data.DataListLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=True,
    pin_memory=True,
  )
  test_loaders = [ hc.data.DataListLoader(
    test_dataset,
    batch_size=torch.cuda.device_count(),
    num_workers=6,
    shuffle=False,
    pin_memory=True,
    ) for test_dataset in test_datasets
  ]

  for e in range(args.warmstart, args.epochs):
    if not args.testing:
      train_result = train(model, train_loader, optimizer, lr_scheduler, args, e)
    test_results = {}
    for test_loader in test_loaders:
      name = test_loader.dataset.name
      print('testing {}'.format(name))
      test_results[name] = test(model, test_loader, args, e)
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
