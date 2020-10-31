import numpy as np
import time
import open3d as o3d

def evaluate(model, correspondences, hyperparams,
             saved_derivatives=None, using_normal=False,
             reweight=False):
  """Construct Optimization Problem.
  Args:
    model: SMPL Model Object.
    correspondences: defined above.
    hyperparams: hyperparameters, interpreted as
      0: weightNN,
      1: weightFeat,
      2: weightPoint2Plane,
      3: weightReg,
      4: lamb,
    saved_derivatives: if is not None, use the saved derivatives

  Returns:
    value_dict: values
    derivative_dict: derivatives w.r.t. params
    hyper_dict: derivatives w.r.t. hyperparams

  """
  # initialization
  joint_weights = hyperparams[:24]
  weightNN = hyperparams[24]
  weightReg = hyperparams[25]
  lamb = hyperparams[26]
  params = np.concatenate([model.trans, model.beta, model.pose], axis=0)
  n_hyper = hyperparams.shape[0]

  # initialize dictionaries
  derivative_dict = {}
  value_dict = {}
  hyper_dict = {}
  derivative_dict['FuncVal'] = np.zeros(85)
  #hyper_dict['FuncVal'] = n_hyper
  value_dict['FuncVal'] = 0.0

  # optimization variables
  Diag = np.eye(params.shape[0])
  Diag[3:13, 3:13] *= 10.0
  value_dict['Q'] = lamb*Diag+weightReg*np.eye(params.shape[0])
  value_dict['b'] = -weightReg*params
  derivative_dict['Q'] = np.zeros((85, 85, 85))
  derivative_dict['b'] = np.zeros((85, 85))
  hyper_dict['Q'] = np.zeros((85, 85, n_hyper))
  hyper_dict['b'] = np.zeros((85, n_hyper))
  hyper_dict['Q'][:, :, 25] = np.eye(params.shape[0]) # weightReg
  hyper_dict['b'][:, 25] = -params
  hyper_dict['Q'][:, :, 26] = Diag # lamb
  value_dict['loss3d'] = np.nan

  """
    0.5*weightReg*||params+dparams||^2
    0.5*weightReg*dparams^2 + weightReg*<params, dparams>
  """
  derivative_dict['FuncVal'] += weightReg*params
  value_dict['FuncVal'] += 0.5*weightReg*np.linalg.norm(params, 2)**2

  if saved_derivatives is not None:
    derivatives = saved_derivatives
  else:
    derivatives = model.compute_derivatives()

  if len(correspondences['joint_indices3d']) > 0:
    joint_correspondences(model, hyperparams, correspondences,
                          derivatives, value_dict, derivative_dict, hyper_dict,
                          reweight=reweight)

  if len(correspondences['indices3d']) > 0:
    point_correspondences(model, hyperparams, correspondences,
                          derivatives, value_dict, derivative_dict, hyper_dict,
                          using_normal=using_normal)

  value_dict['dparams'] = np.linalg.pinv(value_dict['Q']).dot(value_dict['b'])
  value_dict['Qinv'] = np.linalg.pinv(value_dict['Q'])
  hyper_dict['dparams'] = np.zeros((85, n_hyper))
  hyper_dict['Qinv'] = np.zeros((85, 85, n_hyper))
  for i in range(n_hyper):
    dQ = hyper_dict['Q'][:, :, i]
    dQinv = -value_dict['Qinv'].dot(dQ).dot(value_dict['Qinv'])
    hyper_dict['Qinv'][:, :, i] = dQinv
    hyper_dict['dparams'][:, i] = (dQinv.dot(value_dict['b'])+
                                   value_dict['Qinv'].dot(hyper_dict['b'][:, i])
                                   )
  #hyper_dict['grad_squared_norm'] = 2.0*hyper_dict['dparams'].T.dot(value_dict['dparams'])
  #test_dict = {'Q': Q, 'b': b, 'Qinv': Qinv, 'dparams': dparams,
  #             'joint_Q3d': joint_Q3d, 'FuncVal': FuncVal,
  #             'grad_squared_norm': np.linalg.norm(dparams, 2)**2}
  #else:
  #  hyper_derivatives = None
  #  test_dict = None

  return value_dict, derivative_dict, hyper_dict

def point_correspondences(model, hyperparams,
                          correspondences, derivatives,
                          value_dict, derivative_dict, hyper_dict,
                          using_normal=False):
  """Establish Point Correspondences.

  """

  """ 3D point correspondences, point-to-plane distance """
  weightNN = hyperparams[24]
  indices3d = correspondences['indices3d'].astype(np.int32)
  source3d_pool = model.verts
  normals = compute_vertex_normals(model)
  normal3d = normals[indices3d, :]
  source3d = source3d_pool[indices3d, :]

  target3d = correspondences['target3d']
  #visualize_points([source3d, target3d])
  J3d_sq = derivatives['v'][indices3d, :, :]
  # [N, 85] = sum([N, 3, 85]* [N, 3, 1])
  # [N*3, 85]
  J3d = J3d_sq.reshape((-1, 85))
  # [N] = sum([N, 3] * [N, 3])

  W3d = np.kron(correspondences['weights3d'], np.ones(3))

  J3dTW = np.multiply(J3d.T, W3d)
  Q3d_point = J3dTW.dot(J3d)
  diff3d = (target3d-source3d).reshape(-1)
  b3d_point = J3dTW.dot(diff3d) # [N*3, 85].T.dot([N*3, N*3]).dot([N*3])
  J3d_plane = np.sum(np.multiply(J3d_sq, normal3d[:, :, np.newaxis]), axis=1)
  target_plane = np.sum(target3d*normal3d, axis=1)
  source_plane = np.sum(source3d*normal3d, axis=1)
  W3d_plane = correspondences['weights3d']
  J3d_planeTW = np.multiply(J3d_plane.T, W3d_plane)
  Q3d_plane = J3d_planeTW.dot(J3d_plane)
  diff_plane = target_plane-source_plane
  b3d_plane = J3d_planeTW.dot(diff_plane.reshape(-1))
  if using_normal:
    Q3d = Q3d_point + Q3d_plane
    b3d = b3d_point + b3d_plane
  else:
    Q3d = Q3d_point
    b3d = b3d_point
  #print('Q3d.norm=%f' % np.linalg.norm(Q3d, 'fro'))
  value_dict['Q'] += Q3d*weightNN
  value_dict['b'] += b3d*weightNN
  value_dict['loss3d'] = np.mean(np.abs(diff_plane))
  #FuncVal += 0.5*diff_plane.reshape(-1).T.dot(W3d_plane).dot(diff_plane.reshape(-1))#*weightPoint2Plane*weightNN
  value_dict['FuncVal'] += 0.5*np.multiply(diff3d, W3d).dot(diff3d)*weightNN
  derivative_dict['FuncVal'] += -b3d_point*weightNN
  hyper_dict['Q'][:, :, 24] = Q3d # weightNN
  hyper_dict['b'][:, 24] = b3d # weightNN
  #dQ_dhyper[:, :, 2] = weightNN*(Q3d_plane-Q3d_point) # weightPoint2Plane
  #db_dhyper[:, 2] = weightNN*(b3d_plane-b3d_point) # weightPoint2Plane

def joint_correspondences(model, hyperparams,
                          correspondences, derivatives,
                          value_dict, derivative_dict, hyper_dict, reweight=False):
  weightFeat = hyperparams[27]
  """ 3D joint correspondences """
  joint_W3d = np.kron(np.array(hyperparams[:24]), np.ones(3))
  joint_indices3d = correspondences['joint_indices3d'].astype(np.int32)
  joint_source3d = model.J[joint_indices3d, :]
  joint_target3d = correspondences['joint_target3d']
  #if visualize:
  #  visualize_points([model.verts, raw_pc, connections(joint_source3d, joint_target3d)])
  joint_J3d = derivatives['J']
  joint_J3d = joint_J3d[joint_indices3d, :, :].reshape((-1, 85))
  if reweight:
    dists = []
    for i in range(24):
      dist_i = np.linalg.norm(model.J[i, :] - correspondences['joint_target3d'][i, :], 2)
      dists.append(dist_i)
    sigma = np.median(dists) #sorted(dists)[10]
    for i in range(24):
      wi = sigma**4/(sigma**4 + dists[i]**4)
      for ii in range(i*3, i*3+3):
        joint_W3d[ii] = wi
      #print(i, dists[i], wi)

  #joint_W3d = np.kron(correspondences['joint_weights3d'], np.ones(3))
  joint_J3dTW = np.multiply(joint_J3d.T, joint_W3d)
  joint_Q3d = joint_J3dTW.dot(joint_J3d)
  joint_diff3d = (joint_target3d-joint_source3d).reshape(-1)
  #import ipdb; ipdb.set_trace()
  joint_b3d = joint_J3dTW.dot(joint_diff3d)
  #stats['func'] = 0.5*joint_diff3d.reshape(-1).dot(joint_W3d).dot(joint_diff3d.reshape(-1))
  #dparams_3d = np.linalg.solve(joint_Q3d, joint_b3d)
  #print('dparams3d=%f' % np.linalg.norm(dparams_3d, 2))
  loss3d = np.mean(np.square(np.linalg.norm(joint_diff3d.reshape((-1, 3)), 2, axis=-1)))
  value_dict['Q'] += joint_Q3d*weightFeat
  value_dict['b'] += joint_b3d*weightFeat
  #print('joint_Q3d.norm=%f' % np.linalg.norm(joint_Q3d, 'fro'))

  value_dict['FuncVal'] += 0.5*np.multiply(joint_diff3d, joint_W3d).dot(joint_diff3d)*weightFeat
  derivative_dict['FuncVal'] += -joint_b3d*weightFeat
  for i in range(24):
    for ii in range(i*3, i*3+3):
      hyper_dict['Q'][:, :, i] += np.outer(joint_J3d[ii, :], joint_J3d[ii, :]) # joint weights
      #hyper_dict['Q'][:, :, :24] = joint_Q3d # weightFeat
      hyper_dict['b'][:, i] += joint_J3d[ii, :]*joint_diff3d[ii] # joint_weights

def compute_vertex_normals(model):
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(model.verts)
  mesh.triangles = o3d.utility.Vector3iVector(model.faces)
  mesh.compute_vertex_normals()
  return np.array(mesh.vertex_normals)
