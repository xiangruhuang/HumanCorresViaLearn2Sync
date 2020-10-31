import sys, os
from human_corres.config import PATH_TO_DATA
from geop.geometry.camera import PinholeCamera
from geop import linalg
from human_corres.utils import visualization as vis
from human_corres.utils import helper
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import open3d as o3d
import glob

def computeGraph(N, faces, knn):
  G = np.zeros((N, N))
  for i in range(faces.shape[0]):
    i1 = faces[i, 0]
    i2 = faces[i, 1]
    i3 = faces[i, 2]
    G[i1, i2] = 1.0
    G[i1, i3] = 1.0
    G[i2, i1] = 1.0
    G[i2, i3] = 1.0
    G[i3, i2] = 1.0
    G[i3, i1] = 1.0
  G0 = np.array(G)
  G_sum = np.eye(N)
  for i in range(knn):
    G_sum = G_sum + G
    G = G.dot(G0)
  edges = [[] for i in range(N)]
  indices = np.where(G_sum > 0.5)
  for i, j in zip(indices[0], indices[1]):
    edges[i].append(j)
  return edges

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="""Render SURREAL Mesh into scans
             from multiple view points""")
  parser.add_argument('--offset', type=int, default=0)
  parser.add_argument('--knn', type=int, default=7)
  parser.add_argument('--length', type=int, default=1000)
  parser.add_argument('--num_views', type=int, default=20)
  args = parser.parse_args()

  PARAMS = '{}/surreal/surreal_smpl_params.mat'.format(PATH_TO_DATA)
  smpl_params = sio.loadmat(PARAMS)['params']
  offset = args.offset
  length = args.length
  #smpl_params = smpl_params[offset:(offset+length), :]
  camera = PinholeCamera()
  #rotation_path = '{}/surreal/render_rotations'.format(PATH_TO_DATA)

  #os.system('mkdir -p %s' % rotation_path)
  #n_views=args.num_views
  #if not os.path.exists('{}/{}.txt'.format(rotation_path, n_views-1)):
  #  thetas = np.linspace(0, np.pi*2, n_views)
  #  #rotations = [linalg.rodriguez(np.random.randn(3)) for i in range(n_views)]
  #  #rotations = [linalg.rodriguez(np.array([1.,0.,0.])*thetas[i] + np.random.randn(3)*0.2) for i in range(n_views)]
  #  for i, rotation in enumerate(rotations):
  #    np.savetxt('{}/{}.txt'.format(rotation_path, i), rotation)
  #else:
  #  rotations = [np.loadtxt('{}/{}.txt'.format(rotation_path, i)).reshape((3, 3)) for i in range(n_views)]

  render_path = '{}/surreal/scans'.format(PATH_TO_DATA)
  MAT_PATH = '{}/{{0:06d}}'.format(render_path)
  MAT = '{}/{{0:06d}}/{{1:03d}}.mat'.format(render_path)
  OBJ = '{}/{{0:06d}}/{{1:03d}}.obj'.format(render_path)
  os.system('mkdir -p %s' % render_path)
  models = helper.loadSMPLModels()
  gt_dsc = helper.loadSMPLDescriptors(desc='Laplacian_n')
  edges = computeGraph(6890, models[0].faces, knn=args.knn)
  for mesh_id in range(offset, offset+length):
    os.system('mkdir -p %s' % MAT_PATH.format(mesh_id))
    params = np.array(smpl_params[mesh_id, :])
    params[11:14] = 0.0
    gender = int(params[0])
    model = models[gender]
    zero_params = np.zeros(85)
    model.update_params(zero_params)
    rest_mesh = vis.getTriangleMesh(model.verts, model.faces)
    params = np.concatenate([np.zeros(3), params[1:]], axis=0)
    model.update_params(params)
    mesh = vis.getTriangleMesh(model.verts, model.faces)
    point_rotations, point_translations = helper.computeLocalRotations(np.array(mesh.vertices), np.array(mesh.triangles), np.array(rest_mesh.vertices), np.arange(np.array(mesh.vertices).shape[0]), edges=edges) # [N, 3]

    #mat_file = MAT.format(mesh_id)
    depths = []
    points3d = []
    valid_pixel_indices = []
    correspondences = []
    intrinsics = []
    extrinsics = []
    params_list = []
    gt_feats = []
    gt_transformations = []
    normed_adjs = []
    current = 0

    #rotations = [linalg.rodriguez(np.random.randn(3)) for i in range(n_views)]
    for i, rotation in enumerate(rotations):
      print('\r{0:06d}_{1:03d}'.format(mesh_id, i), end="")
      transformation = np.eye(4)
      transformation[:3, :3] = rotation
      mesh.transform(transformation)
      depth_image, extrinsic, intrinsic, points3d_i, correspondence, valid_idx = camera.project(mesh.vertices, mesh.triangles)
      mesh.transform(transformation.T)
      depth = depth_image[(valid_idx[:, 0], valid_idx[:, 1])]

      #point_rotations_i_temp = np.matmul(point_rotations[correspondence, :, :], rotation.T)
      #point_rotations_i = []
      #point_transformed = []
      #for j in range(points3d_i.shape[0]):
      #  Ri = point_rotations_i_temp[j]
      #  ri = linalg.rot2axis_angle(Ri)
      #  point_rotations_i.append(ri)
      #point_rotations_i = np.array(point_rotations_i)
      #point_translations_i = point_translations[correspondence, :]
      ##helper.save_to_obj(OBJ.format(mesh_id, i), vis.getPointCloud(points3d_i))
      #point_transformations_i = np.concatenate([point_rotations_i, point_translations_i], axis=-1).astype(np.float32)
      mat_file = MAT.format(mesh_id, i)
      sio.savemat(mat_file, {'depths': depth.astype(np.float32),
                             'points3d': points3d_i.astype(np.float32),
                             'valid_pixel_indices': valid_idx.astype(np.float32),
                             'correspondences': correspondence,
                             'rotations': rotation.astype(np.float32),
                             'width': 320, 'height': 240,
                             'params': np.array(smpl_params[mesh_id, :]).astype(np.float32),
                             'intrinsics': intrinsic.astype(np.float32),
                             'extrinsics': extrinsic.astype(np.float32),
                             #'gt_transformations': point_transformations_i,
                            }, do_compression=True)
