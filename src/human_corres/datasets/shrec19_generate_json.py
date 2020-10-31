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
           Generate JSON files for mesh rendering of SHREC19-human
           """)
parser.add_argument('--PATH_TO_DATA',
  default='../data/SHREC19/mesh',
  help='the path to where / folder is placed'
)
parser.add_argument('--PATH_TO_JSON',
  default='../data/json/shrec19_rendering.json',
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
mesh_files = []
for i in range(1, 45):
  mesh_files.append(osp.join(args.PATH_TO_DATA, '{}.ply'.format(i)))

json_dict['mesh_files'] = mesh_files
mat_file_paths = [f.replace('.ply', '/{:03d}.mat').replace('mesh', 'scans')
                  for f in mesh_files]
json_dict['mat_file_paths'] = mat_file_paths

""" Store Correspondence as a vertex-wise attribute for training. """
correspondences = []
print(mesh_files)
for cfile in [mesh_file.replace('.ply', '.corres') 
              for mesh_file in mesh_files]:
  correspondence = np.loadtxt(cfile).astype(np.int32).tolist()
  correspondences.append(correspondence)
#json_dict['correspondence_files'] = correspondences
json_dict['meshv_attr_dict'] = { }
json_dict['meshv_list_attr_dict'] = { 'correspondence': correspondences }

""" Dumping JSON file """
with open(args.PATH_TO_JSON, 'w') as fout:
  json.dump(json_dict, fout)
