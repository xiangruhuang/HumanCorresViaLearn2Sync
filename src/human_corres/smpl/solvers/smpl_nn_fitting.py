import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from geop.geometry import util as geo_util
from geop import linalg
import open3d as o3d
from . import smpl_derivatives

def connection(p1, p2):
  points = np.array([(i/1000.0)*p1 + (1.0-i/1000.0)*p2 for i in range(1001)])
  return points

def connections(p1s, p2s):
  n = p1s.shape[0]
  points = []
  for i in range(n):
    p1 = p1s[i, :]
    p2 = p2s[i, :]
    points.append(np.array([(i/1000.0)*p1 + (1.0-i/1000.0)*p2 for i in range(1001)]))
  points = np.concatenate(points, axis=0)
  return points

def visualize_points(points_list):
  import open3d as o3d
  pcds = []
  for points in points_list:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #colors = np.zeros((points.shape[0], 3))
    #colors[:, 0] = np.arange(points.shape[0]) / points.shape[0]
    #colors[:, 1] = np.arange(points.shape[0]) / points.shape[0]
    #colors[:, 2] = np.arange(points.shape[0]) / points.shape[0]
    #pcd.colors = o3d.utility.Vector3dVector(colors)
    pcds.append(pcd)
  pcds[0].paint_uniform_color([1.0, 0, 0])
  pcds[1].paint_uniform_color([0.0, 1.0, 0])
  if len(pcds) > 2:
    pcds[2].paint_uniform_color([0.0, 0.0, 1.0])
  o3d.draw_geometries(pcds)

def update_nn_correspondences(correspondences, model, raw_pc,
                              max_nn_dist, nc_3d, visualize=False):
  w = 0.001
  pcd = o3d.geometry.PointCloud()
  print(raw_pc.shape)
  pcd.points = o3d.utility.Vector3dVector(raw_pc)
  o3d.estimate_normals(pcd)
  signs = np.sign(np.array(pcd.normals).dot(np.array([1.0,0,0])))
  pcd.normals = o3d.utility.Vector3dVector(signs[:, np.newaxis]*np.array(pcd.normals))
  #o3d.io.write_point_cloud('hey.ply', pcd)
  #dirc = np.array([[a*1.0,0,0] for a in range(1000)])/100.0
  #pcd0 = o3d.geometry.PointCloud()
  #pcd0.points = o3d.utility.Vector3dVector(dirc)
  #o3d.draw_geometries([pcd, pcd0])

  raw = np.concatenate([raw_pc, w*np.array(pcd.normals)], axis=1)
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(model.verts)
  mesh.triangles = o3d.utility.Vector3iVector(model.faces)
  mesh.compute_vertex_normals()
  model_v = np.concatenate([model.verts, w*np.array(mesh.vertex_normals)], axis=1)

  """
  Compute nearest neighbor correspondences
  from current mesh vertices to point cloud.
  """
  tree = NN(n_neighbors=1).fit(model_v)
  dists, indices = tree.kneighbors(raw)
  indices = indices[:, 0]
  idx = np.where(dists < max_nn_dist)[0]
  nnmask = idx
  indices = indices[idx]
  dists = np.abs(np.sum(np.multiply(raw_pc[idx, :]-model_v[indices, :3], model_v[indices, 3:]), axis=1))
  dists = np.square(dists)
  argmax = np.argmax(dists)
  nn_vertices = raw_pc[idx, :]
  #nn_vertices = model.verts[indices[idx], :]
  nn_vec = np.array(nn_vertices).reshape(-1)
  sigma2 = np.median(dists)
  #print('sigma2=%f' % sigma2)
  nn_weights=np.ones(dists.shape) # (np.ones(dists.shape)*sigma2/(np.ones(dists.shape)*sigma2+dists)) #np.exp(-dists/(2*sigma2))
  if visualize:
    #print(dists[argmax])
    #print(nn_weights[argmax])
    #import ipdb; ipdb.set_trace()
    visualize_points([model.verts, raw_pc, connection(raw_pc[idx[argmax], :], model_v[indices[argmax], :3])])

  indices3d = correspondences['indices3d'][:nc_3d]
  target3d = correspondences['target3d'][:nc_3d].reshape((-1, 3))
  weights3d = correspondences['weights3d'][:nc_3d]
  indices3d = np.concatenate([indices3d, indices], axis=0)
  target3d = np.concatenate([target3d, raw_pc], axis=0)
  weights3d = np.concatenate([weights3d, nn_weights], axis=0)
  correspondences['indices3d'] = indices3d
  correspondences['target3d'] = target3d
  correspondences['weights3d'] = weights3d

def update_nn2d_correspondences(correspondences, model, raw, max_nn2d_dist):
  """
  Compute nearest neighbor correspondences
  from current mesh vertices to point cloud.
  """
  raw_pixels = linalg.pointcloud2pixel(raw, model.camera[1], model.camera[0])
  v_pixels = linalg.pointcloud2pixel(model.verts,
                                     model.camera[1], model.camera[0])
  tree = NN(n_neighbors=1).fit(raw_pixels)
  dists, indices = tree.kneighbors(v_pixels)
  dists = dists[:, 0]
  indices = indices[:, 0]
  idx = np.where(dists < max_nn2d_dist)[0]
  nnmask = idx
  nn_vertices = raw[indices[idx], :]
  nn_vec = np.array(nn_vertices).reshape(-1)
  return nnmask, nn_vec

def projection2d(model, points):
  """ project points to 2d pixels w.r.t. model.camera
  Args:
    points: [n, 3] np.ndarray representing points.
    model: SMPL model.

  Returns:
    pixels: [n, 2] np.ndarray representing projected pixels.
  """
  intrinsic, extrinsic = model.camera
  R, trans = geo_util.unpack(extrinsic)
  pixels = intrinsic.dot(R.dot(points.T)+trans.reshape((3, 1))).T
  pixels[:, 0] = pixels[:, 0] / pixels[:, 2]
  pixels[:, 1] = pixels[:, 1] / pixels[:, 2]
  return pixels[:, :2]

def update_model(model, params):
  """Change model deformation parameters to ``params''.

  """
  model.set_params(trans=params[:3],
                   beta=params[3:13],
                   pose=params[13:])

#def test_derivatives(model, correspondences, hyperparams):
#  return_dict0 = construct_derivatives(
#    model, correspondences, hyperparams, compute_derivatives=True)
#  dparams0 = return_dict0['dparams']
#  hyper_derivatives0 = return_dict0['hyper_derivatives']
#  test_dict0 = return_dict0['test_dict']
#  gradient = return_dict0['dF_dparams']
#  eps = 1e-8
#
#  params = np.concatenate([model.trans, model.beta, model.pose], axis=0)
#  FuncVal0 = return_dict0['FuncVal']
#  for i in range(params.shape[0]):
#    params[i] += eps
#    update_model(model, params)
#    return_dict = construct_derivatives(
#      model, correspondences, hyperparams, compute_derivatives=True)
#    #assert np.linalg.norm(return_dict['source_plane'] - (FuncVal*normal3d).sum(1)) < 1e-6
#    #assert np.linalg.norm(return_dict['J3d_plane'][:, :, i] - J3d_plane[:, :, i]) < 1e-6
#    FuncVal = return_dict['FuncVal']
#    emp = (FuncVal-FuncVal0)/eps
#    theory = gradient[i]
#    if np.linalg.norm(emp-theory)/np.linalg.norm(emp) > 1e-2:
#      import ipdb; ipdb.set_trace()
#      print('emp=%f, theory=%f' % (np.linalg.norm(emp, 2),
#                                           np.linalg.norm(theory, 2)))
#    params[i] -= eps
#  update_model(model, params)
#
#  for i in range(hyperparams.shape[0]):
#    if i == 2:
#      continue
#    dhyper = np.zeros(hyperparams.shape[0])
#    dhyper[i] = 1.0
#    hyperparams += dhyper*eps
#    return_dict = construct_derivatives(
#      model, correspondences, hyperparams, compute_derivatives=True)
#    dparams = return_dict['dparams']
#    test_dict = return_dict['test_dict']
#    for key in hyper_derivatives0.keys():
#      print(key)
#      if key == 'b' and i == 4:
#        # numerical unstable, ignore.
#        continue
#      emp = np.array((test_dict[key]-test_dict0[key])/eps)
#      theory = hyper_derivatives0[key].dot(dhyper)
#      if np.linalg.norm(emp-theory)/np.linalg.norm(emp) > 1e-2:
#        import ipdb; ipdb.set_trace()
#        print('key=%s, emp=%f, theory=%f' % (key, np.linalg.norm(emp, 2),
#                                             np.linalg.norm(theory, 2)))
#    hyperparams -= dhyper*eps

def smpl_fitting(correspondences, model, raw_pc, hyperparams,
                 args, using_nn_counter=30, decay=0.5,
                 using_normal=False, max_iter=100, reweight=False):
  """Fitting an SMPL model with correspondences.

  Additionally, adding dynamic nearest neighbor correspondences.
  Consider a bipartite graph G(U, V, E). Left vertex set U represents
  SMPL 3D joint/vertex locations and 2D joint/vertex locations, in
  total (6890+24)*2 nodes. Right vertex set V represents handles,
  including 3D points and 2D pixels. Edge set E represents weighted
  connections between U and V, each edge represents a correspondence.
  U is given naturally by the SMPLModel and V is from raw input data.
  Edge set E consists of two parts: I) from precomputed (thus fixed)
  correspondences. II) from dynamic nearest neighbor computation.

  Args:
    correspondences: a dictionary containing all fixed correspondences.
      i.e. representing the fixed subset of edge set V.
      nodes in U are numbered in the following way:
      In 'indices3d' and 'target3d':
        [0-6890): vertices on SMPL mesh template.

      In 'joint_indices3d' and 'joint_target3d':
        [0, 24): joints of SMPL model.

      In 'indices2d' and 'target2d':
        [0-6890): 2D projection of vertices on SMPL mesh
          template. Projection is performed w.r.t. camera intrinsics
          and extrinsics stored in model.camera (of type tuple).
        [6890-6890+24): 2D projection of joints of SMPL model,
          projected the same way as mesh vertices.

      Assuming nc correspondences are given, we have the dictionary:
      {
      'indices2d': [nc_2d] integers ([] indicating no correspondence),
          each represent a node in U.
      'indices3d': [nc_3d] integers ([] indicating no correspondence),
          each represent a node in U.
      'joint_indices3d': [24] integers ([] indicating no correspondence),
          each represent a node in U.

      'target2d': [nc_2d, 2] target 2D pixels, representing the
          2D subset of V.
      'target3d': [nc_3d, 3] target 3D locations, representing the
          3D subset of V.
      'joint_target3d': [24, 3] target 3D locations, representing the
          3D subset of V.
      'weights2d': [nc_2d] weights for 2D correspondences.
      'weights3d': [nc_3d] weights for 3D correspondences.
      'joint_weights3d': [24] weights for 3D correspondences.
      } where
      nc_2d: number of correspondences to pixels.
      nc_3d: number of correspondences to 3D points.
    model: a SMPLModel object. Initialized with intrinsic shape
      and pose blend directions.
    raw_pc: a o3d.geometry.PointCloud object, representing raw point cloud.
    max_nn_dist: a float. threshold for valid nearest neighbor correspondence.
    max_iter: a integer. maximum number of iterations
    lamb: a float. Gauss-Newton second order parameters
    sigma: a float. soft threshdold for reweighting.

  Returns:
    params: vector of shape [85], representing translation(3)+shape(10)+pose(72).
    stats: statistics.
  """
  #import ipdb; ipdb.set_trace()
  params = np.concatenate([model.trans, model.beta, model.pose.reshape(-1)], axis=0)
  params_list = []
  params_list.append(params)
  update_model(model, params)
  nc_2d = correspondences['indices2d'].shape[0]
  nc_3d = correspondences['indices3d'].shape[0]
  stats = {'loss3d': [], 'loss3d_joint': [],
           'loss2d': [],
           'dparams': 0.0}
  using_nn = False
  print('nc2d=%d, nc3d=%d' % (nc_2d, nc_3d))
  for i in range(max_iter):
    if i == using_nn_counter:
      using_nn = True
    #  visualize_points([model.verts, raw_pc])
    """ Get correspondences and weights """
    if using_nn:
      update_nn_correspondences(correspondences, model, raw_pc,
                                args.max_nn_dist, nc_3d, ((i+1)%10==0))
    #if using_nn and (i % 10 == 0):
    #  hyperparams[27] *= decay

    """Update Params via optimization
      x === params
      jv === concat([vec(smpl_joints), vec(smpl_vertices)])
      target_jv = concat([vec(target_joints), vec(target_vertices)])

      min_{dx} ||J dx + jv - target_jv||^2 + lambda || dx ||^2 gives
      min_{dx} <dx, Q dx> + <b, dx>
      Q = (J^T W J + lambda I)
      b = J^TW(target_jv-jv)
    """

    values, derivatives, hyper_derivatives = smpl_derivatives.evaluate(
                                               model, correspondences,
                                               hyperparams,
                                               using_normal=using_normal,
                                               reweight=reweight)
    dparams = values['dparams']

    params = np.array(params + dparams)
    update_model(model, params)

    output_string = '\riter=%d, std_3d=%f, delta params=%f'
    print(output_string % (i, values['loss3d'],
                           np.linalg.norm(dparams, 2)),
          end="")
    params_list.append(np.array(params))
    if np.linalg.norm(dparams, 2) < 3e-3:
      break
    #if np.linalg.norm(dparams, 2) < args.stopping:
    #  break
  values, derivatives, hyper_derivatives = smpl_derivatives.evaluate(
                                             model, correspondences,
                                             hyperparams)
  print('')
  return { 'params': params, 'params_list': params_list, 'loss': values['loss3d']}

def print_hyperparameters(hyperparams):
  output_string = 'weightNN=%(0)f, weightFeat=%(1)f, weightPoint2Plane=%(2)f, weightReg=%(3)f, lamb=%(4)f'
  print(output_string % {str(i): hp for i, hp in enumerate(hyperparams)})

#def loss_surface_optimization(correspondences, model, raw_pc,
#                              hyperparams, args, params_list):
#  print('#sample points=%d' % len(params_list))
#  nc_3d = correspondences['indices3d'].shape[0]
#  update_nn_correspondences(correspondences, model, raw_pc,
#                            args.max_nn_dist, nc_3d)
#  stats = {'loss3d': [], 'loss3d_joint': [],
#           'loss2d': [],
#           'dparams': 0.0}
#  gt_params = np.concatenate([model.trans, model.beta, model.pose], axis=0)
#  for i in range(100):
#    loss = 0.0
#    dhyperparams = np.zeros(hyperparams.shape[0])
#    for params in params_list:
#      update_model(model, params)
#      dparams, hyper_derivatives, test_dict = construct_derivatives(
#                      model, correspondences, stats,
#                      hyperparams, compute_derivatives=True)
#      loss += np.linalg.norm(dparams-(gt_params - params), 2)**2
#      dhyperparams += hyper_derivatives['dparams'].T.dot(dparams-(gt_params - params))
#      print('.', end="")
#    print_hyperparameters(hyperparams)
#    hyperparams -= dhyperparams
#    print('loss=%f' % loss)

