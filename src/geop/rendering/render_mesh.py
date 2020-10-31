import argparse
import json

from geop.geometry.camera import PinholeCamera
from geop import linalg
#from hybrid_corres.utils import visualization as vis
#from hybrid_corres.utils import helper
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import os, os.path as osp
import open3d as o3d

class MeshRendering:
  def __init__(self, json_file, depth_image=False):
    with open(json_file, 'r') as fin:
      json_dict = json.load(fin)
    self.camera = PinholeCamera()
    self.rotations = np.array(json_dict['rotations'])
    print('\tTotal #view={}'.format(self.rotations.shape[0]))
    self.mesh_files = json_dict['mesh_files']
    print('\tTotal #mesh={}'.format(len(self.mesh_files)))
    self.mat_file_paths = json_dict['mat_file_paths']
    self.depth_image = depth_image
    if json_dict.get('meshv_attr_dict', None) is not None:
      self.meshv_attr_dict = json_dict['meshv_attr_dict']
    if json_dict.get('meshv_list_attr_dict', None) is not None:
      self.meshv_list_attr_dict = json_dict['meshv_list_attr_dict']
    #else:
    #  if json_dict.get('correspondence_files', None) is not None:
    #    self.meshv_attr_dict = { 'correspondence': [] }
  
  def render(self, offset, length):
    for mesh_id in range(offset, offset+length):
      print('mesh id = {}'.format(mesh_id))
      mesh_file = self.mesh_files[mesh_id]
      mat_file_path = self.mat_file_paths[mesh_id]
      mesh = o3d.io.read_triangle_mesh(mesh_file)
      
      os.system('mkdir -p {}'.format(osp.dirname(mat_file_path)))
      for view_id, rotation in enumerate(self.rotations):
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        mesh.transform(transformation)
        packed = self.camera.project(mesh.vertices, mesh.triangles)
        depth_image, extrinsic, intrinsic = packed[0], packed[1], packed[2]
        points3d_i, correspondence, valid_idx = packed[3], packed[4], packed[5]
        mesh.transform(transformation.T)
        output_dict = {}
        if self.depth_image:
          depth = depth_image[(valid_idx[:, 0], valid_idx[:, 1])]
          output_dict['depth'] = depth
          output_dict['valid_pixel_indices'] = valid_idx
          output_dict['width'] = 320
          output_dict['height'] = 240
        
        output_dict['points3d'] = points3d_i
        #output_dict['correspondence'] = gt_correspondence
        for name, attr in self.meshv_attr_dict.items():
          output_dict[name] = np.array(attr)[correspondence]
        for name, attr in self.meshv_list_attr_dict.items():
          output_dict[name] = np.array(attr[mesh_id])[correspondence]
        output_dict['mesh_correspondence'] = correspondence
        output_dict['rotation'] = rotation
        output_dict['intrinsic'] = intrinsic
        output_dict['extrinsic'] = extrinsic
        
        mat_file = mat_file_path.format(view_id)
        print('saving to %s' % mat_file)
        sio.savemat(mat_file, output_dict, do_compression=True)
        
       
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="""Render Mesh into Scans""")
  parser.add_argument('--json_file', default=None, type=str,
                      help='see instructions of mesh rendering.')
  #parser.add_argument('--path_to_data', default=None, type=str,
  #                    help='see instructions of mesh rendering.')
  parser.add_argument('--offset', type=int, default=0,
                      help='starting index [default: 0]')
  parser.add_argument('--length', type=int, default=1,
                      help='number of meshes to render [default: 1]')
  parser.add_argument('--depth_image', action='store_true')
  args = parser.parse_args()
  print('rendering from {} to {}'.format(args.offset, args.offset+args.length))
  render_machine = MeshRendering(args.json_file, args.depth_image) #.path_to_data, args)
  render_machine.render(args.offset, args.length)
