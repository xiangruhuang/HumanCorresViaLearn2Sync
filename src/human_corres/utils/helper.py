import open3d as o3d
import numpy as np
import os, sys
import os.path as osp
from human_corres.smpl import SMPLModel
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import human_corres.utils.visualization as vis
from geop import linalg
from human_corres.config import PATH_TO_SMPLDESC, PATH_TO_SMALDESC
import human_corres as hc
import glob

def pose_error(gt_rotations, rotations, gt_translations, translations):
  """
  Args:
    gt_rotations: [A1, A2, ..., 3, 3]
    rotations: [A1, A2, ..., 3, 3]
    gt_translations: [A1, A2, ..., 3]
    translations: [A1, A2, ..., 3]
  Returns:
    aerrs: angular errors in degrees (360) [A1, A2, ...]
    terrs: translation errors [A1, A2, ...]
  """
  RR = tf.matmul(rotations, gt_rotations,
                 transpose_b=True)
  cost = tf.clip_by_value((tf.reduce_sum(tf.linalg.diag_part(RR), axis=-1)-1.0)/2.0, -1.0, 1.0)
  aerrs = tf.math.acos(cost) / np.pi * 180.0
  terrs = tf.sqrt(tf.reduce_sum(tf.square(gt_translations - translations), axis=-1))
  return aerrs, terrs

def depth2image(depth, points, indices, height, width):
  image = np.zeros((height, width, 5))
  data = np.zeros((points.shape[0], 5))
  data[:, 0] = depth
  data[:, 1] = 1.0
  data[:, 2:] = points
  index_tuples = tuple([indices[:, 0], indices[:, 1]])
  image[index_tuples] = data
  return image

def save_to_obj(filename, pcd):
  """
  Args:
    filename: string.
    pcd: o3d.geometry.PointCloud.
  """
  points = np.array(pcd.points)
  with open(filename, 'w') as fout:
    for i in range(points.shape[0]):
      fout.write('v %.6f %.6f %.6f\n' % (points[i, 0],
                                         points[i, 1],
                                         points[i, 2]))

def write_obj(filename, points):
  """
  Args:
    filename: string.
    pcd: o3d.geometry.PointCloud.
  """
  with open(filename, 'w') as fout:
    for i in range(points.shape[0]):
      fout.write('v %.6f %.6f %.6f\n' % (points[i, 0],
                                         points[i, 1],
                                         points[i, 2]))

def read_obj(filename):
  """
  Args:
    filename: string.
  Returns:
    mesh: o3d.geometry.TriangleMesh.
  """
  points = []
  triangles = []
  with open(filename, 'r') as fin:
    for line in fin.readlines():
      if len(line.strip()) < 2:
        continue
      if line.startswith('v '):
        points.append([float(token) for token in line.strip().split(' ')[1:4]])
      if line.startswith('f '):
        triangles.append([int(token) for token in line.strip().split(' ')[1:4]])
  if len(triangles) > 0:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles).astype(np.int32))
    return mesh
  else:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    return pcd

def read_SHREC19_pairs():
  pairs = []
  gt_indices_list = []
  pairs_txt = glob.glob(osp.join(hc.PATH_TO_SHREC19, 'pairs_gt', '*.txt'))
  for txt_file in pairs_txt:
    pair = txt_file.split('/')[-1].split('.')[0]
    i, j = [int(token)-1 for token in pair.split('_')]
    gt_indices = np.loadtxt(txt_file).astype(np.int32) - 1
    pairs.append((i, j))
    gt_indices_list.append(gt_indices)
  return pairs, gt_indices_list

def read_FAUST_intra_pairs():
  pairs = []
  gt_indices_list = []
  MAT = osp.join(hc.PATH_TO_FAUST_SCANS, 'eval_pairs.mat')
  mat = sio.loadmat(MAT)
  pairs = mat['intra']
  gt_indices_list = [np.arange(6890) for i in range(50)]
  return pairs, gt_indices_list

def read_FAUST_inter_pairs():
  pairs = []
  gt_indices_list = []
  MAT = osp.join(hc.PATH_TO_FAUST_SCANS, 'eval_pairs.mat')
  mat = sio.loadmat(MAT)
  pairs = mat['inter']
  gt_indices_list = [np.arange(6890) for i in range(50)]
  return pairs, gt_indices_list

def read_FAUST_Test_intra_pairs():
  pairs = []
  gt_indices_list = []
  MAT = osp.join(hc.PATH_TO_FAUST_TEST_SCANS, 'eval_pairs.mat')
  mat = sio.loadmat(MAT)
  pairs = mat['intra'].astype(np.int32)
  gt_indices_list = [np.arange(6890) for i in range(pairs.shape[0])]
  return pairs, gt_indices_list

def read_FAUST_Test_inter_pairs():
  pairs = []
  gt_indices_list = []
  MAT = osp.join(hc.PATH_TO_FAUST_TEST_SCANS, 'eval_pairs.mat')
  mat = sio.loadmat(MAT)
  pairs = mat['inter'].astype(np.int32)
  gt_indices_list = [np.arange(6890) for i in range(pairs.shape[0])]
  return pairs, gt_indices_list

def loadSMPLModels():
  smpl_models = {}
  smpl_models[0] = SMPLModel(osp.join(PATH_TO_SMPLDESC, 'model_intrinsics/female_model.pkl'))
  smpl_models[1] = SMPLModel(osp.join(PATH_TO_SMPLDESC, 'model_intrinsics/male_model.pkl'))
  return smpl_models

def loadSMALModels():
  smpl_models = {}
  smpl_models['horse'] = o3d.io.read_triangle_mesh(osp.join(PATH_TO_SMALDESC, 'model_intrinsics/horse0.ply'))
  smpl_models['cat'] = o3d.io.read_triangle_mesh(osp.join(PATH_TO_SMALDESC, 'model_intrinsics/cat0.ply'))
  return smpl_models

def loadSMPLDescriptors(desc):
  mat = sio.loadmat(osp.join(PATH_TO_SMPLDESC, 'embedding/descriptors.mat'))
  desc = mat[desc]
  desc = (desc - desc.min(0)[np.newaxis, :])/(desc.max(0)-desc.min(0))[np.newaxis, :]
  return desc

def loadSMALDescriptors(desc):
  mat = sio.loadmat(osp.join(PATH_TO_SMALDESC, 'embedding/animals.mat'))
  desc = mat[desc]
  desc = (desc - desc.min(0)[np.newaxis, :])/(desc.max(0)-desc.min(0))[np.newaxis, :]
  return desc

def simpleICP(p, q):
  def updateT(p, q, R):
    cp = np.mean(p, axis=0)
    cq = np.mean(q, axis=0)
    trans = cq - R.dot(cp.T).T
    return trans
  def updateR(p, q, t = np.zeros(3)):
    cp = np.mean(p, axis=0)
    cq = np.mean(q, axis=0)
    cq = cq - t.reshape((1, 3))
    dp = p - cp
    dq = q - cq
    M = dp.T.dot(dq)
    U, S, VT = np.linalg.svd(M)
    Sign = np.eye(3)
    R = VT.T.dot(Sign).dot(U.T)
    if np.linalg.det(R) < 0:
      Sign[2, 2] = -1
      R = VT.T.dot(Sign).dot(U.T)
    return R
  n = p.shape[0]
  R = updateR(p, q)
  t = updateT(p, q, R)
  #for itr in range(10):
  #  loss = np.linalg.norm(R.dot(p.T).T + t.reshape((1, 3)) - q, 'fro') ** 2
  #  t = updateT(p, q, R)
  #  R = updateR(p, q, t)
  #  print('iter=%d, loss=%f' % (itr, loss))
  return R, t

def reweightICP(p, q, stopping=0.1):
  def updateT(p, q, w, R):
    cp = (p * w[:, np.newaxis]).sum(axis=0)/w.sum()
    cq = (q * w[:, np.newaxis]).sum(axis=0)/w.sum()
    trans = cq - R.dot(cp.T).T
    return trans
  def updateR(p, q, w, t = np.zeros(3)):
    cp = (p * w[:, np.newaxis]).sum(axis=0)/w.sum()
    cq = (q * w[:, np.newaxis]).sum(axis=0)/w.sum()
    cq = cq - t.reshape((1, 3))
    dp = (p - cp)*w[:, np.newaxis]
    dq = q - cq
    M = dp.T.dot(dq)
    U, S, VT = np.linalg.svd(M)
    Sign = np.eye(3)
    R = VT.T.dot(Sign).dot(U.T)
    if np.linalg.det(R) < 0:
      Sign[2, 2] = -1
      R = VT.T.dot(Sign).dot(U.T)
    return R
  decay = 0.95
  eps = 2.0
  n = p.shape[0]
  w = np.ones(n)
  while eps > stopping:
    R = updateR(p, q, w)
    t = updateT(p, q, w, R)
    dists = np.linalg.norm((R.dot(p.T) + t.reshape((3, 1))).T - q, 2, axis=-1)
    w[np.where(dists > eps)] = 0.0
    eps = eps * decay

  #for itr in range(10):
  #  loss = np.linalg.norm(R.dot(p.T).T + t.reshape((1, 3)) - q, 'fro') ** 2
  #  t = updateT(p, q, R)
  #  R = updateR(p, q, t)
  #  print('iter=%d, loss=%f' % (itr, loss))
  return R, t

def computeLocalRotations(src_points, faces, tgt_points, correspondences, edges=None):
  #tree = NN(n_neighbors=knn, n_jobs=10).fit(src_points)
  #dists, indices = tree.kneighbors(src_points)
  N = src_points.shape[0]
  rotations = np.zeros((N, 3, 3))
  translations = np.zeros((N, 3))

  for i in range(N):
    neighbors = []
    for e in edges[i]:
      neighbors.append(e)
    neighbors = np.array(neighbors).astype(np.int32)
    Ri, ti = simpleICP(src_points[neighbors, :], tgt_points[correspondences[neighbors], :])
    ##v1 = Ri.dot(src_points[neighbors, :].T).T + ti.reshape((1, 3))
    ##v2 = tgt_points[correspondences[neighbors], :]
    ##pcd1 = vis.getPointCloud(v1)
    ##pcd1.paint_uniform_color([1,0,0])
    ##pcd2 = vis.getPointCloud(v2)
    ##pcd2.paint_uniform_color([0,1,0])
    ##o3d.draw_geometries([pcd1, pcd2])

    #pi = src_points[i, :].reshape((1, 3)) # [1, 3]
    ##dists_i = dists[i, :]
    ##valid_idx = np.where(dists_i < 0.05)[0]
    ##neighbors = indices[i, 1:]
    #ci = correspondences[i]
    #cj = correspondences[neighbors]
    #idx = np.where(cj != ci)[0]
    #if len(idx) < 3:
    #  import ipdb; ipdb.set_trace()
    #  assert False
    #pj = src_points[neighbors, :][idx, :] # [10, 3]
    #qi = tgt_points[correspondences[i], :].reshape((1, 3))
    #qj = tgt_points[correspondences[neighbors], :][idx, :]
    #M = (pi - pj).T.dot(qi-qj) # [3, 10] * [10, 3]
    #U, S, VT = np.linalg.svd(M)
    #Sign = np.eye(3)
    #R = VT.T.dot(Sign).dot(U.T)
    #if np.linalg.det(R) < 0:
    #  Sign[2, 2] = -1
    #  R = VT.T.dot(Sign).dot(U.T)
    rotations[i, :] = Ri
    translations[i, :] = ti
    #dist = (pi-pj).dot(R.T) - (qi - qj)
    #e1 = vis.getEdges(pi-np.zeros((10, 3)), pj)
    #e2 = vis.getEdges(qi-np.zeros((10, 3)), qj)
    #import ipdb; ipdb.set_trace()
    #o3d.draw_geometries([e1, e2])
    #pi = R.dot(pi.T).T
    #pj = R.dot(pj.T).T
    #e1 = vis.getEdges(pi-np.zeros((10, 3)), pj)
    #e2 = vis.getEdges(qi-np.zeros((10, 3)), qj)
    #o3d.draw_geometries([e1, e2])
    #
    #print(np.linalg.norm(dist))

  #for i in range(N):
  #  neighbors = []
  #  m = np.zeros(3)
  #  for e in edges[i]:
  #    Rie = rotations[i].dot(rotations[e].T)
  #    rie = linalg.rot2axis_angle(M)
  #    m += rie
  #  m /= len(edges)
  #  distortion[i, :] = m

  return rotations, translations

def parse_args():
  def exit_with_help():
    print('Error: --mat must be set.')
    print('try python data_parser.py -h')
    exit(-1)

  import argparse
  parser = argparse.ArgumentParser('fit smpl to raw 3d point cloud')

  parser.add_argument('--mat', type=str, default='examples/input.mat',
                      help='.mat data file to load')
  parser.add_argument('--n_period', type=int, default=1,
                      help='number of loss surfaces to optimize')
  parser.add_argument('--lamb', type=float, default=1.0,
                      help='parameter for Gauss-Newton [default 1.0]')
  parser.add_argument('--wsmooth', type=float, default=0.0,)
  parser.add_argument('--wline', type=float, default=0.0,)
  parser.add_argument('--walign', type=float, default=0.0,)
  parser.add_argument('--wball', type=float, default=10.0,)
  parser.add_argument('--max_nn_dist', type=float, default=100,
                      help='maximum nearest neighbor distance [default 100]')
  parser.add_argument('--weightPoint2Plane', type=float, default=0.0,
                      help='weight of point-to-plane distance in (0, 1), [default 0.5]')
  parser.add_argument('--weightFeat', type=float, default=100.0,
                      help='weight of coarse feature matching [default 1.0]')
  parser.add_argument('--weightNN', type=float, default=1.0,
                      help='weight of Nearest Neighbor correspondences \
                            matching [default 1.0]')
  parser.add_argument('--weightReg', type=float, default=0.1,
                      help='weight of Regularization [default 1.0]')
  parser.add_argument('--stopping', type=float, default=1e-2,
                      help='stopping condition of norm(dparams) [default 1e-2]')
  parser.add_argument('--sigma', type=float, default=1.0,
                      help='soft threshold for reweighting [default 1.0]')
  parser.add_argument('--max_iter', type=int, default=50,
                      help='maximum number of iterations [default 50]')
  parser.add_argument('--smpl_intrinsics', type=str, default='model_intrinsics',
                      help='folder of SMPL intrinsic parameters \
                            containing male.mat, female.mat \
                            [default "model_intrinsics"]')

  args = parser.parse_args()
  if args.mat is None:
    exit_with_help()
  return args

def visualize_vertices(vertices, points):
  """
  Visualize vertices given by a [N, 3] np.ndarray.

  """
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(vertices)
  pcd.paint_uniform_color([0,1,0])
  pcd1 = o3d.geometry.PointCloud()
  pcd1.points = o3d.utility.Vector3dVector(points)
  pcd1.paint_uniform_color([0,0,1])

  o3d.draw_geometries([pcd, pcd1])

def save_to_ply(points, filename, colors=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  o3d.estimate_normals(pcd)
  if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.io.write_point_cloud(filename, pcd)

def save_mesh_to_ply(points, triangles, filename, colors=None):
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(points)
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  mesh.compute_vertex_normals()
  o3d.io.write_triangle_mesh(filename, mesh)

def save_convergence(losses, filename, title):
  import matplotlib.pyplot as plt
  plt.plot(range(len(losses)), losses)
  plt.yscale('log')
  plt.xlabel('Iteration')
  plt.ylabel('Sum of Squared Distances')
  plt.title(title)
  plt.savefig(filename)

def joint_names():
  return ['hips', #0
          'leftUpLeg',
          'rightUpLeg',
          'spine',
          'leftLeg', #4
          'rightLeg',
          'spine1',
          'leftFoot',
          'rightFoot',
          'spine2', #9
          'leftToeBase',
          'rightToeBase',
          'neck',
          'leftShoulder',
          'rightShoulder', #14
          'head',
          'leftArm',
          'rightArm',
          'leftForeArm',
          'rightForeArm', #19
          'leftHand',
          'rightHand',
          'leftHandIndex1',
          'rightHandIndex1']

def visualize_points(points):
  import open3d as o3d
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  o3d.draw_geometries([pcd])

def visualize_skeleton(joints, kintree_table, filename):
  fig = plt.figure()
  plt.scatter(joints[:, 0], joints[:, 1], 1.0, 'r')
  for i in range(1, kintree_table.shape[1]):
    parent = kintree_table[0, i]
    plt.plot([joints[i, 0], joints[parent, 0]],
             [joints[i, 1], joints[parent, 1]],
             'b')
  plt.xlim([0, 320])
  plt.ylim([0, 240])
  plt.savefig(filename)
  plt.close(fig)

def joints2skeleton(joints3d, kintree_table, filename):
  additions = []
  num_interp = 50
  for i in range(1, kintree_table.shape[1]):
    if i in [10, 11]:
      continue
    parent = kintree_table[0, i]
    x = joints3d[i, :]
    y = joints3d[parent, :]
    for j in range(num_interp):
      ratio = j*1.0/(num_interp-1)
      mid = x*(1-ratio) + ratio*y
      additions.append(mid)
  pcd = o3d.geometry.PointCloud()
  additions = np.stack(additions, axis=0)
  print(additions.shape, joints3d.shape)
  pcd.points = o3d.utility.Vector3dVector(np.concatenate([joints3d, additions], axis=0))
  o3d.io.write_point_cloud(filename, pcd)

def feature_averaging(points, features, rotations, target_points):
  """Average point features.

  Args:
    points: [N, N_points, 3], N point clouds.
    features: [N, N_points, d], point-wise features for each point cloud.
    rotations: [N, 3, 3], rotation of each point cloud.
    target_points: [M, 3], points to average features on.

  Returns:
    target_features: [M, d] averaged features.
  """
  N = points.shape[0]
  attached_points = []
  for i in range(N):
    points_i = points[i, :, :]
    Ri = rotations[i]
    points_i = Ri.T.dot(points_i.T).T
    attached_points.append(points_i)
  attached_points = np.concatenate(attached_points, axis=0) # [N*N_points, 3]
  attached_features = features.reshape((-1, features.shape[-1])) # [N*N_points, d]
  tree = NN(n_neighbors=1).fit(attached_points)
  dists, indices = tree.kneighbors(target_points) # [M, 5]
  extracted_features = attached_features[indices, :] # [M, 5, d]
  average_features = extracted_features.mean(axis=1) # [M, d]
  return average_features

def masked_feature_averaging(points, features, rotations, target_points):
  """Average point features.

  Args:
    points: [N, N_points, 3], N point clouds.
    features: [N, N_points, d], point-wise features for each point cloud.
    rotations: [N, 3, 3], rotation of each point cloud.
    target_points: [M, 3], points to average features on.

  Returns:
    mask: [X] of range [0, 6890)
    target_features: [M, d] averaged features.
  """
  N = points.shape[0]
  attached_points = []
  for i in range(N):
    points_i = points[i, :, :]
    Ri = rotations[i]
    points_i = Ri.T.dot(points_i.T).T
    attached_points.append(points_i)
  attached_points = np.concatenate(attached_points, axis=0) # [N*N_points, 3]
  attached_features = features.reshape((-1, features.shape[-1])) # [N*N_points, d]
  tree = NN(n_neighbors=5).fit(attached_points)
  dists, indices = tree.kneighbors(target_points) # [M, 5]
  extracted_features = attached_features[indices, :] # [M, 5, d]
  average_features = extracted_features.mean(axis=1) # [M, d]
  mask = np.where(dists[:, 0] < 0.02)[0]
  return mask, average_features[mask, :]

def predict_smpl_correspondence(features, sigma=1e-2):
  """Compute feature correspondence by Nearest Neighboring.

  Args:
    features: [M, d] point-wise features.

  Returns:
    correspondence: [M] integers, each in [0, 6890).
    confidence: [M] floats, indicating a confidence score in [0, 1].
  """
  dsc_dim = features.shape[-1]
  dsc = loadSMPLDescriptors()['male'][:, :dsc_dim]
  tree = NN(n_neighbors=1, n_jobs=10).fit(dsc)
  dists, indices = tree.kneighbors(features)
  indices = indices[:, 0]
  dists = dists[:, 0]
  sigma2 = sigma*sigma
  confidence = sigma2/(dists + sigma2)
  correspondence = indices
  return correspondence, confidence

def compute_smpl_correspondence_scores(features):
  """Compute feature correspondence by Nearest Neighboring.

  Args:
    features: [N, d] point-wise features.

  Returns:
    correspondence_scores: [N, M] floats.
  """
  dsc_dim = features.shape[-1]
  dsc = loadSMPLDescriptors()['male'][:, :dsc_dim]
  diff = (features[:, np.newaxis, :] - dsc[np.newaxis, :, :])
  diff_norms = np.linalg.norm(diff, 2, axis=-1) # [N, M]
  return -diff_norms

def normed_adj(points):
  tree = NN(n_neighbors=5).fit(points)
  dists, indices = tree.kneighbors(points)
  neighbors = [[] for i in range(points.shape[0])]
  N = points.shape[0]
  adj = np.zeros((N, N))
  for i in range(N):
    for dist, j in zip(dists[i, :], indices[i, :]):
      if np.linalg.norm(points[i, :]-points[j, :], 2) < 0.1:
        neighbors[i].append(j)
        neighbors[j].append(i)

  for i in range(N):
    neighbors[i] = list(set(neighbors[i]))
  
  geo_dist = np.zeros((N, N)) + 1e10
  visited = np.zeros(N).astype(np.int32)
  for i in range(N):
    queue = []
    for j in neighbors[i]:
      queue.append(j)
      geo_dist[i, j] = np.linalg.norm(points[i, :] - points[j, :], 2)
      visited[j] = 1
    l = 0
    while l < len(queue):
      j = queue[l]
      for k in neighbors[j]:
        if (geo_dist[i, j] + geo_dist[j, k] < geo_dist[i, k]):
          geo_dist[i, k] = geo_dist[i, j] + geo_dist[j, k]
        if visited[k] == 0:
          if geo_dist[i, k] < 0.1:
            visited[k] = 1
            queue.append(k)
      l += 1
    #neighbors[i] = queue
    for j in queue:
      visited[j] = 0
      adj[i, j] = 1.0
  normed_adj = adj / adj.sum(axis=-1)[:, np.newaxis]
  return normed_adj
