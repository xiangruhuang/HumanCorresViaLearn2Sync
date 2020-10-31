import argparse
import torch
import human_corres as hc
from human_corres.config import PATH_TO_CHECKPOINT
import os, os.path as osp
import glob

def construct_animal_datasets(args, mesh=False):
  train_dataset = hc.datasets.SMALFEPts(
    descriptor_dim=args.embed_dim,
    split=args.train_split,
    desc='laplacian',
  )
  test_datasets = []
  test_datasets.append(
    hc.datasets.SMALFEPts(
      descriptor_dim=args.embed_dim,
      split=args.test_split,
      desc='laplacian',
    )
  )

  return train_dataset, test_datasets

def construct_datasets(args, mesh=False):
  train_dataset = hc.datasets.SurrealFEPts(
    descriptor_dim=args.embed_dim,
    split=args.train_split,
    desc=args.desc,
    cls=args.cls,
  )
  test_datasets = []
  test_datasets.append(
    hc.datasets.Shrec19FEPts(
      descriptor_dim=args.embed_dim,
      split=args.test_split,
      cls=args.cls,
      desc=args.desc,
    )
  )
  test_datasets.append(
    hc.datasets.FaustFEPts(
      descriptor_dim=args.embed_dim,
      split=args.test_split,
      cls=args.cls,
      desc=args.desc,
    )
  )
  #test_datasets.append(
  #  hc.datasets.SurrealFEPts(
  #    descriptor_dim=args.embed_dim,
  #    split=args.test_split,
  #    cls=args.cls,
  #    desc=args.desc,
  #  )
  #)
  #test_datasets.append(
  #  hc.datasets.FaustTestFEPts(
  #    descriptor_dim=args.embed_dim,
  #    split=args.test_split,
  #    cls=args.cls,
  #    desc=args.desc,
  #  )
  #)

  if mesh:
    mesh_datasets = []
    mesh_datasets.append(
      hc.datasets.Shrec19Mesh(
        split=args.test_split,
      )
    )
    mesh_datasets.append(
      hc.datasets.FaustMesh(
        split=args.test_split,
      )
    )
    mesh_datasets.append(
      hc.datasets.SurrealMesh(
        split=args.test_split,
      )
    )
    return train_dataset, test_datasets, mesh_datasets
  else:
    return train_dataset, test_datasets

def parse_args():
  parser = argparse.ArgumentParser(description=
    """
    Dense Human Correspondence Network with Transformation Synchronization.
    """)
  parser.add_argument(
    "--batch_size", type=int, default=10,
    help="Batch size [default: 10]"
  )
  parser.add_argument(
    "--warmstart", type=str, default='0',
    help="""
         Indicating which checkpoint to load. 
         If set to 'latest', will look for the most recent checkpoint.
         The actual .pt file to be loaded is
         <hybrid_corres.config.PATH_TO_CHECKPOINT>/
           <args.checkpoints>/phase<phase>/<args.warmstart>.pt
         phase is determined by condition (args.transf_reg).
         [default: '0']
         """,
  )
  parser.add_argument(
    "--checkpoints", type=str, default='TEMP',
    help="""
         Checkpoints will be saved in a subfolder of
         folder `hybrid_corres.config.PATH_TO_CHECKPOINT`.
         with folder name=`args.checkpoints`, [default: 'TEMP']
         """,
  )
  parser.add_argument(
    "--embed_dim", type=int, default=100,
    help="""
         Dimensionality of Laplacian Embedding, must not be larger than 128
         [default: 100].
         """,
  )
  parser.add_argument(
    "--animals", action='store_true',
    help="if True, train on animal shapes",
  )
  parser.add_argument(
    "--cpu", action='store_true',
    help="if True, train with CPU only",
  )
  parser.add_argument(
    "--desc", type=str, default='Laplacian_n',
    help="""
         which descriptor to use for feature extraction supervision.
         (see function loadSMPLDescriptors in hybrid_corres.utils.helper.py)
         Candidates for human shapes include
         {Laplacian_n, HKS, WKS, SIHKS}.
         Candidates for animal shapes include {laplacian}. 
         [default: 'Laplacian_n']
         """
  )
  parser.add_argument(
    "--weight_decay", type=float, default=1e-6,
    help="L2 regularization coeff [default: 1e-6]",
  )
  parser.add_argument(
    "--lr", type=float, default=1e-3,
    help="Initial learning rate [default: 1e-3]"
  )
  parser.add_argument(
    "--lr_clip", type=float, default=1e-5,
    help="Stopping learning rate [default: 1e-5]"
  )
  parser.add_argument(
    "--lr_decay", type=float, default=0.9,
    help="Learning rate decay gamma [default: 0.9]",
  )
  parser.add_argument(
    "--decay_step", type=float, default=100000,
    help="Learning rate decay step [default: 1e5]",
  )
  parser.add_argument(
    "--bn_momentum", type=float, default=0.9,
    help="Initial batch norm momentum [default: 0.9]",
  )
  parser.add_argument(
    "--bn_decay", type=float, default=0.9,
    help="Batch norm momentum decay gamma [default: 0.9]",
  )
  parser.add_argument(
    "--epochs", type=int, default=200,
    help="Number of epochs to train [default: 200]"
  )
  parser.add_argument(
    "--testing", action='store_true',
    help="If true, test only."
  )
  parser.add_argument(
    "--parallel_period", type=int, default=0,
    help="""
         If set to larger than 0, only work on a subset of data according to
         condition (id % parallel_period == parallel_split).
         """
  )
  parser.add_argument(
    "--parallel_split", type=int, default=0,
    help="""
         If parallel_period is set to larger than 0,
         only work on a subset of data according to
         condition (id % parallel_period == parallel_split).
         """
  )
  parser.add_argument(
    "--val", action='store_true',
    help="If true, use validation set only."
  )
  parser.add_argument(
    "--FEnet", type=str, default='PointNet2',
    help="""
         network architecture for feature extraction, 
         see hybrid_corres.modules.pointnet2
         [default: PointNet2]""",
  )
  parser.add_argument(
    "--RegNet", type=str, default='Reg',
    help="""
         network arch for transformation regularization
         see hybrid_corres.modules.reg
         """,
  )
  parser.add_argument(
    "--dump_result", action='store_true',
    help="if True, dump result. [default: False]",
  )
  parser.add_argument(
    "--transf_reg", action='store_true',
    help="if True, use transformation regularization module [default: False]",
  )
  parser.add_argument(
    "--init", action='store_true',
    help="if True, initialize transformation regularization module [default: False]",
  )
  args = parser.parse_args()
  args.device = torch.device('cuda:0' if (torch.cuda.is_available() and (not args.cpu)) else 'cpu')
  args.ckpt_path = osp.join(PATH_TO_CHECKPOINT, args.checkpoints)
  args.cls = False
  args.train_split = 'train'
  args.test_split = 'test'
  if args.val:
    args.train_split = 'val'
    args.test_split = 'val'
  if args.transf_reg:
    args.phase = 2
  else:
    args.phase = 1
  args.save_path = osp.join(args.ckpt_path, 'phase{}'.format(args.phase))
  os.system('mkdir -p {}'.format(args.save_path))
  args.load_path = osp.join(args.ckpt_path, 'phase{}'.format(args.phase))
  if args.warmstart == 'latest':
    checkpoints = glob.glob(osp.join(args.load_path, '*.pt'))
    if args.transf_reg and (len(checkpoints) == 0):
      args.phase_transition = True
      args.load_path = osp.join(args.ckpt_path, 'phase{}'.format(1))
      checkpoints = glob.glob(osp.join(args.load_path, '*.pt'))
    else:
      args.phase_transition = False
    latest_timestamp = -1e10
    latest_checkpoint = -1
    for checkpoint in checkpoints:
      t = osp.getmtime(checkpoint)
      if latest_timestamp < t:
        latest_timestamp = t
        latest_checkpoint = osp.basename(checkpoint).split('.')[0]
    args.warmstart = int(latest_checkpoint) + 1
  else:
    args.warmstart = int(args.warmstart)
  return args

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


