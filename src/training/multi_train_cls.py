import os, os.path as osp
import re
import glob
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
#import torch_geometric.transforms as T
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn_interpolate
import torch.optim.lr_scheduler as lr_sched
import progressbar
import argparse
import hybrid_corres as hc
from hybrid_corres.utils import helper
from hybrid_corres.config import PATH_TO_CHECKPOINT
from hybrid_corres.nn import DataParallel
from training.model import HybridModel

def correspondence(x, y):
  """
  Args:
    x: [N, D]
    y: [M, D]
  Returns:
    corres: [N] integers.
  """
  x2 = (x**2).sum(-1).view(x.shape[0], 1)
  y2 = (y**2).sum(-1).view(1, y.shape[0])
  xydist = x2 + y2 - 2.0*torch.mm(x, y.transpose(0, 1))
  corres = xydist.argmin(dim=-1)
  return corres

def parse_args():
  parser = argparse.ArgumentParser(description="Arg parser")
  parser.add_argument(
    "--batch_size",
    type=int,
    default=10,
    help="Batch size [default: 10]"
  )
  parser.add_argument(
    "--num_points",
    type=int,
    default=8192,
    help="Number of points to train with [default: 8192]",
  )
  parser.add_argument(
    "--gpu",
    type=str,
    default='0',
    help="GPU ID to use [default: 0, 1]",
  )
  parser.add_argument(
    "--warmstart",
    type=str,
    default='0',
    help="which checkpoint to load [default: '0']",
  )
  parser.add_argument(
    "--checkpoints",
    type=str,
    default='TEMP',
    help="""
         Checkpoints will be saved in a subfolder of
         folder `hybrid_corres.config.PATH_TO_CHECKPOINT`.
         with folder name=`checkpoints`, [default ID: 'TEMP']
         """,
  )
  parser.add_argument(
    "--embed_dim",
    type=int,
    default=100,
    help="Dimensionality of Laplacian Embedding [default: 100]",
  )
  parser.add_argument(
    "--weight_decay",
    type=float,
    default=0,
    help="L2 regularization coeff [default: 0]",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Initial learning rate [default: 1e-3]"
  )
  parser.add_argument(
    "--lr_clip", type=float, default=1e-5,
    help="Stopping learning rate [default: 1e-5]"
  )
  parser.add_argument(
    "--lr_decay",
    type=float,
    default=0.9,
    help="Learning rate decay gamma [default: 0.9]",
  )
  parser.add_argument(
    "--decay_step",
    type=float,
    default=100000,
    help="Learning rate decay step [default: 1e5]",
  )
  parser.add_argument(
    "--bn_momentum",
    type=float,
    default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
  )
  parser.add_argument(
    "--bn_decay",
    type=float,
    default=0.9,
    help="Batch norm momentum decay gamma [default: 0.9]",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="Number of epochs to train for"
  )
  parser.add_argument(
    "--testing",
    type=bool,
    default=False,
    help="If true, test only."
  )
  parser.add_argument(
    "--val",
    type=bool,
    default=False,
    help="If true, use validation set only."
  )

  parser.add_argument(
    "--FEnet",
    type=str,
    default='PointNet2',
    help="network arch for feature extraction",
  )
  parser.add_argument(
    "--pc_sampler",
    type=str,
    default='uniform5000',
    help="""
         Point Cloud Sampler, {uniform5000, voxel0.02}, [default: uniform5000]
         """,
  )
  parser.add_argument(
    "--build_graph",
    type=str,
    default='radius',
    help="""
         tools to build graph {knn, radius}, [default: 'radius']
         """,
  )
  args = parser.parse_args()
  args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  args.ckpt_path = osp.join(PATH_TO_CHECKPOINT, args.checkpoints)
  args.cls = True
  args.train_split = 'train'
  args.test_split = 'test'
  if args.val:
    args.train_split = 'val'
    args.test_split = 'val'
  if not osp.exists(args.ckpt_path):
    os.system('mkdir -p %s' % args.ckpt_path)
  with open(osp.join(args.ckpt_path, 'params.table'), 'w') as fout:
    for key in dir(args):
      if key.startswith('_'):
        continue
      item = getattr(args, key)
      fout.write('%20s \t %40s\n' % (key, str(item)))
  if args.warmstart == 'latest':
    checkpoints = glob.glob(osp.join(args.ckpt_path, '*.pt'))
    latest_timestamp = -1e10
    latest_checkpoint = 0
    for checkpoint in checkpoints:
      t = osp.getmtime(checkpoint)
      if latest_timestamp < t:
        latest_timestamp = t
        latest_checkpoint = osp.basename(checkpoint).split('.')[0]
    args.warmstart = int(latest_checkpoint) + 1
  else:
    args.warmstart = int(args.warmstart)
  return args

def train(model, loader, optimizer, lr_scheduler, args, epoch):
  model.train()

  total_l2dist = total_loss = 0.0
  widgets = [
    progressbar.ETA(), ' training, ',
    progressbar.Variable('avg_loss'), ', ',
    progressbar.Variable('avg_l2dist'), ', ',
    progressbar.Variable('count'), ', ',
    progressbar.Variable('lr'), ', ',
    progressbar.Variable('i'), ', ',
  ]
  template_points = torch.Tensor(loader.dataset.template_points).to(args.device)
  #template_feats = torch.Tensor(loader.dataset.template_feats).to(args.device)

  with progressbar.ProgressBar(max_value=len(loader), widgets=widgets) as bar:
    for i, data_list in enumerate(loader):
      optimizer.zero_grad()
      out = model(data_list)
      y = torch.cat([data.y for data in data_list]).to(out.device)
      loss = F.nll_loss(out, y)
      loss.backward()
      optimizer.step()
      lr_scheduler.step()
      with torch.no_grad():
        pred = out.max(1)[1]
        l2dist = (((template_points[pred, :] - template_points[y, :]) ** 2).sum(-1)
                  .sqrt().mean())
        total_l2dist += l2dist.item()
        total_loss += loss.item()
      if ((i + 1) % 1000 == 0) or (i == len(loader)-1):
        bar.update(
          i,
          avg_loss=total_loss/(i+1.0),
          avg_l2dist=total_l2dist/(i+1.0),
          count=(i+1)*args.batch_size,
          lr=lr_scheduler.get_lr()[0],
          i=i,
        )
        torch.save({
          'epoch': epoch,
          'afteriter': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'lr_scheduler_state_dict': lr_scheduler.state_dict(),
          'avg_loss': total_loss / (i+1.0),
          'avg_l2dist': total_l2dist / (i+1.0),
          }, osp.join(args.ckpt_path, '%d.pt' % (epoch))
        )
  torch.cuda.empty_cache()
  return { 'avg_loss': total_loss / len(loader),
           'avg_l2dist': total_l2dist / len(loader) }

def test(model, loader, args, epoch):
  model.eval()

  total_loss = total_l2dist = 0.0
  widgets = [
    progressbar.ETA(), ' testing, ',
    progressbar.Variable('avg_loss'), ', ',
    progressbar.Variable('avg_l2dist'), ', ',
    progressbar.Variable('count'), ', ',
  ]
  template_points = torch.Tensor(loader.dataset.template_points).to(args.device)
  #template_feats = torch.Tensor(loader.dataset.template_feats).to(args.device)

  with torch.no_grad():
    with progressbar.ProgressBar(max_value=len(loader), widgets=widgets) as bar:
      for i, data_list in enumerate(loader):
        out = model(data_list)
        y = torch.cat([data.y for data in data_list]).to(out.device)
        loss = F.nll_loss(out, y)
        with torch.no_grad():
          pred = out.max(1)[1]
          l2dist = (((template_points[pred, :] - template_points[y, :]) ** 2).sum(-1)
                    .sqrt().mean())
          total_l2dist += l2dist.item()
          total_loss += loss.item()
        if (i % 1000 == 0) or (i == len(loader)-1):
          bar.update(
            i,
            avg_loss=total_loss/(i+1.0),
            avg_l2dist=total_l2dist/(i+1.0),
            count=(i+1)*args.batch_size,
          )

  torch.cuda.empty_cache()
  return { 'avg_l2dist': total_l2dist / len(loader),
           'avg_loss': total_loss / len(loader) }

if __name__ == '__main__':
  args = parse_args()
  print(args)
  torch.cuda.manual_seed_all(816)
  model = HybridModel(args)
  if args.FEnet in ['GraphSAGE', 'GAT', 'GCN', 'GraphUNet']:
    train_dataset = hc.datasets.SurrealFEPts(
      descriptor_dim=args.embed_dim,
      sampler=args.pc_sampler,
      split=args.train_split,
      build_graph=args.build_graph,
      cls=True,
    )
    test_datasets = []
    test_datasets.append(
      hc.datasets.FaustFEPts(
        descriptor_dim=args.embed_dim,
        sampler=args.pc_sampler,
        split=args.test_split,
        build_graph=args.build_graph,
        cls=True,
      )
    )
    test_datasets.append(
      hc.datasets.SurrealFEPts(
        descriptor_dim=args.embed_dim,
        sampler=args.pc_sampler,
        split=args.test_split,
        build_graph=args.build_graph,
        cls=True,
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
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

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
      save_dict['train_loss'] = train_result['avg_loss']
      for name, test_result in test_results.items():
        save_dict['test_l2dist_{}'.format(name)] = test_result
      torch.save(
        save_dict,
        osp.join(args.ckpt_path, '%d.pt' % (e))
      )
    else:
      break
