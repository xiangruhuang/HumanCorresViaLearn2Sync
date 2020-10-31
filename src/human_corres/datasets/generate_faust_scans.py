from human_corres.config import PATH_TO_DATA
from geop.geometry.camera import PinholeCamera
from geop import linalg
from human_corres.utils import visualization as vis
from human_corres.utils import helper
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import os

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="""Render FAUST Mesh into scans
             from multiple view points""")
  parser.add_argument('--num_views', type=int, default=100,
                      help='how many rendering views [default: 100]')
  parser.add_argument('--offset', type=int, default=0,
                      help='starting index [default: 0]')
  parser.add_argument('--length', type=int, default=0,
                      help='number of meshes to render [default: 1]')
  args = parser.parse_args()
  SCAN = '%s/MPI-FAUST/training/scans/tr_scan_{0:03d}.ply' % (PATH_TO_DATA)
  REG = '%s/MPI-FAUST/training/registrations/tr_reg_{0:03d}.ply' % (PATH_TO_DATA)
  FAUST_DESC = '%s/faust/faust_descriptors/{0:03d}.mat' % (PATH_TO_DATA)
  rotation_path = '%s/faust/render_rotations' % (PATH_TO_DATA)
  render_path = '%s/faust/scans' % (PATH_TO_DATA)
  OBJ = '%s/{0:03d}_{1:03d}.obj' % (render_path)
  MESH_CORRES = '%s/MPI-FAUST/training/registrations/tr_reg_{0:03d}.corres' % (PATH_TO_DATA)
  CORRES = '%s/{0:03d}_{1:03d}.corres' % (render_path)
  MAT = '%s/{0:03d}_{1:03d}.mat' % (render_path)

  import open3d as o3d
  import glob
  camera = PinholeCamera()
  #n_points = 3000
  os.system('mkdir -p %s' % rotation_path)
  n_views=100
  if not os.path.exists('%s/%d.txt' % (rotation_path, n_views-1)):
    thetas = np.linspace(0, np.pi*2, n_views)
    rotations = [linalg.rodriguez(np.array([0.,1.,0.])*thetas[i] + np.random.randn(3)*0.2) for i in range(n_views)]
    for i, rotation in enumerate(rotations):
      np.savetxt('%s/%d.txt' % (rotation_path, i), rotation)
  else:
    rotations = [np.loadtxt('%s/%d.txt' % (rotation_path, i)).reshape((3, 3)) for i in range(n_views)]

  os.system('mkdir -p %s' % render_path)
  rest_mesh = helper.loadSMPLModels()[0]
  #edges = computeGraph(6890, rest_mesh.faces, knn=7)
  rest_mesh = vis.getTriangleMesh(rest_mesh.verts, rest_mesh.faces)
  gt_descriptors = helper.loadSMPLDescriptors(desc='Laplacian_n')
  dsc_tree = NN(n_neighbors=1, n_jobs=10).fit(gt_descriptors)
  
  for scan_id in range(args.offset, args.offset+args.length):
    """ Correspondence translation """
    #raw_mesh = o3d.io.read_triangle_mesh(SCAN.format(scan_id)) # raw mesh
    reg_mesh = o3d.io.read_triangle_mesh(REG.format(scan_id)) # registration mesh
    #tree = NN(n_neighbors=1, n_jobs=10).fit(np.array(reg_mesh.vertices))
    #dists, indices = tree.kneighbors(np.array(raw_mesh.vertices))
    #scan2reg = indices[:, 0]
    #Nraw = np.array(raw_mesh.vertices).shape[0]
    Nreg = np.array(reg_mesh.vertices).shape[0]

    mesh = o3d.io.read_triangle_mesh(REG.format(scan_id))
    dsc_file = FAUST_DESC.format(scan_id)
    gt_dsc = sio.loadmat(dsc_file)['dsc'] # [6890, 128]
    _, gt_corres = dsc_tree.kneighbors(gt_dsc) # [6890, 128]
    gt_corres = gt_corres[:, 0]
    np.savetxt(MESH_CORRES.format(scan_id), gt_corres, fmt='%d')
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
    #point_rotations, point_translations = helper.computeLocalRotations(
    #    np.array(mesh.vertices), np.array(mesh.triangles),
    #    np.array(rest_mesh.vertices), np.arange(Nraw),
    #    edges=edges) # [N, 3]
    for i, rotation in enumerate(rotations):
      transformation = np.eye(4)
      transformation[:3, :3] = rotation
      mesh.transform(transformation)
      depth_image, extrinsic, intrinsic, points3d_i, correspondence, valid_idx = camera.project(mesh.vertices, mesh.triangles)
      #correspondence = scan2reg[correspondence]
      mesh.transform(transformation.T)
      depth = depth_image[(valid_idx[:, 0], valid_idx[:, 1])]
      helper.save_to_obj(OBJ.format(scan_id, i), vis.getPointCloud(points3d_i))
      #_, gt_correspondence = dsc_tree.kneighbors(gt_dsc[correspondence, :])
      gt_correspondence = gt_corres[correspondence]
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
