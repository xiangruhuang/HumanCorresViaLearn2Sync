import argparse
import scipy.io as sio
from hybrid_corres.utils import helper
from hybrid_corres.smpl import smpl_align
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

parser = argparse.ArgumentParser()
parser.add_argument('--offset', type=int, default=0)
parser.add_argument('--length', type=int, default=1)
parser.add_argument('--dataset', type=str, default=None)

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
#CORRES='../data/result/{}/{{}}_errors.mat'.format(args.dataset)
ERRORS='../data/result/{}/{{}}_errors.mat'.format(args.dataset)

errors_list = []
for i in range(args.offset, args.offset+args.length):
  errors = sio.loadmat(ERRORS.format(i))['errors_pac'].reshape(-1, 2)
  #errors = np.loadtxt(ERRORS.format(i))
  errors_list.append(errors)
  errors_pac = np.concatenate(errors_list, axis=0)
  #errors_pac 
  if i % 1000 == 0:
    print(errors_pac.mean(0),(errors_pac < 0.05).astype(np.float32).mean(0), (errors_pac < 0.1).astype(np.float32).mean(0))
errors = np.concatenate(errors_list, axis=0)
print(errors_pac.mean(0),(errors_pac < 0.05).astype(np.float32).mean(0), (errors_pac < 0.1).astype(np.float32).mean(0))
