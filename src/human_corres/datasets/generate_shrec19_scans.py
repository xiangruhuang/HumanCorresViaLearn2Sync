from human_corres.config import PATH_TO_DATA
from geop.geometry.camera import PinholeCamera
from geop import linalg
from human_corres.utils import visualization as vis
from human_corres.utils import helper
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import os
import open3d as o3d
import glob
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="""Render SHREC19 Mesh into scans
             from multiple view points""")
  parser.add_argument('--num_views', type=int, default=100,
                      help='how many rendering views [default: 100]')
  parser.add_argument('--offset', type=int, default=0,
                      help='starting index [default: 0]')
  parser.add_argument('--length', type=int, default=0,
                      help='number of meshes to render [default: 44]')
  args = parser.parse_args()
  n_views = args.num_views
  SCAN = '%s/SHREC19/mesh/{}.ply' % (PATH_TO_DATA)
  GTCORRES = '%s/SHREC19/mesh/{}.corres' % (PATH_TO_DATA)
  rotation_path = '%s/SHREC19/render_rotations' % (PATH_TO_DATA)
  render_path = '%s/SHREC19/scans' % (PATH_TO_DATA)
  OBJ = '%s/{0:03d}_{1:03d}.obj' % (render_path)
  CORRES = '%s/{0:03d}_{1:03d}.corres' % (render_path)
  MAT = '%s/{0:03d}_{1:03d}.mat' % (render_path)

  camera = PinholeCamera()
  os.system('mkdir -p %s' % rotation_path)
  if not os.path.exists('%s/%d.txt' % (rotation_path, n_views-1)):
    thetas = np.linspace(0, np.pi*2, n_views)
    rotations = [linalg.rodriguez(np.array([0.,1.,0.])*thetas[i] + np.random.randn(3)*0.2) for i in range(n_views)]
    for i, rotation in enumerate(rotations):
      np.savetxt('%s/%d.txt' % (rotation_path, i), rotation)
  else:
    rotations = [np.loadtxt('%s/%d.txt' % (rotation_path, i)).reshape((3, 3)) for i in range(n_views)]

  os.system('mkdir -p %s' % render_path)
  rest_mesh = helper.loadSMPLModels()[0]
  rest_mesh = vis.getTriangleMesh(rest_mesh.verts, rest_mesh.faces)
  #gt_descriptors = helper.loadSMPLDescriptors('Laplacian_N')['male']
  #dsc_tree = NN(n_neighbors=1, n_jobs=10).fit(gt_descriptors)
  IDlist = np.arange(1, 45)
  
  for scan_id in IDlist[args.offset:(args.offset+args.length)]:
    """ Correspondence translation """
    #raw_mesh = o3d.io.read_triangle_mesh(SCAN.format(scan_id)) # raw mesh
    #reg_mesh = o3d.io.read_triangle_mesh(REG.format(scan_id)) # registration mesh
    #tree = NN(n_neighbors=1, n_jobs=10).fit(np.array(reg_mesh.vertices))
    #dists, indices = tree.kneighbors(np.array(raw_mesh.vertices))
    #scan2reg = indices[:, 0]
    #Nraw = np.array(raw_mesh.vertices).shape[0]
    #Nreg = np.array(reg_mesh.vertices).shape[0]

    mesh = o3d.io.read_triangle_mesh(SCAN.format(scan_id))
    smpl_corres = np.loadtxt(GTCORRES.format(scan_id))
    #dsc_file = FAUST_DESC.format(scan_id)
    #gt_dsc = gt_descriptors[smpl_corres, :]
    #gt_dsc = sio.loadmat(dsc_file)['dsc'] # [6890, 128]
    #gt_dsc = gt_dsc[scan2reg[np.arange(Nraw)], :] # [1m, 128]
    n_views = len(rotations)
    depths = []
    points3d = []
    valid_pixel_indices = []
    correspondences = []
    intrinsics = []
    extrinsics = []
    gt_feats = []
    gt_correspondences = []
    current = 0

    used_rotations = []
    for i, rotation in enumerate(rotations):
      transformation = np.eye(4)
      transformation[:3, :3] = rotation
      mesh.transform(transformation)
      depth_image, extrinsic, intrinsic, points3d_i, correspondence, valid_idx = camera.project(mesh.vertices, mesh.triangles)
      mesh.transform(transformation.T)
      depth = depth_image[(valid_idx[:, 0], valid_idx[:, 1])]
      helper.save_to_obj(OBJ.format(scan_id, i), vis.getPointCloud(points3d_i))
      gt_correspondence = smpl_corres[correspondence]
      np.savetxt(CORRES.format(scan_id, i), gt_correspondence, '%d')
      mat_file = MAT.format(scan_id, i)
      print('saving to %s' % mat_file)
      sio.savemat(mat_file, {'depth': depth, 'points3d': points3d_i,
                             'valid_pixel_indices': valid_idx,
                             'correspondence': gt_correspondence,
                             'mesh_correspondence': correspondence,
                             'rotation': rotation,
                             'width': 320, 'height': 240,
                             'intrinsic': intrinsic,
                             'extrinsic': extrinsic,
                            }, do_compression=True)
