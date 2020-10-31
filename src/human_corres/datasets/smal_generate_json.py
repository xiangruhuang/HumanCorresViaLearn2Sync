import json
import glob
#from human_corres.config import PATH_TO_DATA
from geop.geometry.camera import PinholeCamera
from geop import linalg
#from human_corres.utils import visualization as vis
#from human_corres.utils import helper
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import os, os.path as osp
import argparse

parser = argparse.ArgumentParser(description="""
           Generate JSON files for mesh rendering of SMAL cat and horse
           """)
parser.add_argument('--PATH_TO_DATA',
  default='../data/smal',
  help='the path to where smal/ folder is placed'
)
parser.add_argument('--PATH_TO_JSON',
  default='../data/json/smal_rendering.json',
  help='the path to where json file will be placed'
)
args = parser.parse_args()

json_dict = {}

""" Generate Random Views """
n_views = 100
thetas = np.linspace(0, np.pi*2, n_views)
rotations = [linalg.rodriguez(np.array([0.,1.,0.])*thetas[i] + 
             np.random.randn(3)*0.2).tolist() for i in range(n_views)]
json_dict['rotations'] = rotations

""" Extract Mesh Files and generate their destination depth scans"""
mesh_files = glob.glob(
               osp.join(args.PATH_TO_DATA, '*', '*.ply'),
             )
json_dict['mesh_files'] = mesh_files
mat_file_paths = [f.replace('.ply', '/{:03d}.mat').replace('smal', 'smal_scans')
                  for f in mesh_files]
json_dict['mat_file_paths'] = mat_file_paths

""" Store Correspondence as a vertex-wise attribute for training. """
json_dict['meshv_attr_dict'] = { 'correspondence': np.arange(3889).tolist() }

""" Dumping JSON file """
with open(args.PATH_TO_JSON, 'w') as fout:
  json.dump(json_dict, fout)
