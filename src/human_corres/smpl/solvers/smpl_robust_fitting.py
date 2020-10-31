import numpy as np

def smpl_robust_fitting(correspondences, model,
                        max_iter=100, lamb=10.0, sigma=10.0):
  """
  Fitting an SMPL model to a point cloud, given correspondences.

  Parameter:
  ----------
  correspondence: a dictionary containing all correspondences.
    {
      'joint_mask': [n_jc] integers or [].
                    The source joint ids on SMPL model, each id in [0,24)
      'joint_target': [n_jc, 3] target 3D locations or [].
                      that corresponds to the source joints.
      'point_mask': [n_pc] integers or [].
                    The source point ids on SMPL model, each id in [0,6890)
      'point_target': [n_pc, 3] target 3D locations or [].
                      that corresponds to the source points.
    } where
    n_jc: number of joint correspondences.
    n_pc: number of point correspondences.

  model: a SMPLModel object.
         Initialized with intrinsic shape and pose blend directions.

  max_iter: a integer. maximum number of iterations

  lamb: a float. Gauss-Newton second order parameters

  sigma: a float. soft threshdold for reweighting.

  Return:
  -------
  params: vector of shape [85].
  losses: losses after each iteration.
  """
  params = np.zeros(85) # np.random.randn(85)
  vertices = correspondences['point_target']
  joints = correspondences['joint_target']
  v_vec = np.array(vertices).reshape(-1)
  j_vec = np.array(joints).reshape(-1)
  target_jv = np.concatenate([j_vec, v_vec], axis=0) # concat([joints, vertices])
  weights = np.ones(target_jv.shape[0]//3, dtype=np.float32)
  vmask = correspondences['point_mask']
  jmask = correspondences['joint_mask']

  def update_vertices_and_joints(model, params):
    model.set_params(trans=params[:3],
                     beta=params[3:13],
                     pose=params[13:])
    return model.verts, model.J

  def retrieve_relevant_jv(smpl_joints, smpl_vertices, jmask, vmask):
    jv = np.concatenate([smpl_joints[jmask, :], smpl_vertices[vmask, :]],
                            axis=0).reshape(-1)
    return jv

  def compute_loss(jv, target_jv):
    l2_loss = np.linalg.norm((jv-target_jv).reshape((-1, 3)), 2, axis=-1)
    return np.square(l2_loss)

  losses = []
  for i in range(max_iter):
    smpl_vertices, smpl_joints = update_vertices_and_joints(model, params)
    jv = retrieve_relevant_jv(smpl_joints, smpl_vertices, jmask, vmask)
    grad = model.compute_derivatives()
    # Jacobian matrix of shape [Unknown, 3, 85]
    jacob_V = grad['v'][vmask, :].reshape((-1, params.shape[0]))
    # Jacobian matrix of shape [Unknown, 3, 85]
    jacob_J = grad['J'][jmask, :].reshape((-1, params.shape[0]))
    J = np.concatenate([jacob_J, jacob_V], axis=0) # [(n_jc+n_pc)*3, 85]
    #J = model.compute_derivatives()['v'][label, :].reshape((-1, params.shape[0]))
    """
    x === params
    jv === concat([vec(smpl_joints), vec(smpl_vertices)])
    target_jv = concat([vec(target_joints), vec(target_vertices)])

    min_{dx} ||J dx + jv - target_jv||^2 + lambda || dx ||^2 gives
    min_{dx} <dx, Q dx> + <W b, dx>
    Q = (J^T W J + lambda I)
    b = J^T(target_jv-jv)

    """
    weightsby3 = np.outer(weights, np.ones(3)).reshape(-1)
    J = np.multiply(J, weightsby3[:, np.newaxis])
    Q = J.T.dot(J) + lamb*np.eye(params.shape[0])
    b = J.T.dot(target_jv-jv)
    dparams = np.linalg.solve(Q, b)
    params += dparams
    smpl_vertices, smpl_joints = update_vertices_and_joints(model, params)
    jv = retrieve_relevant_jv(smpl_joints, smpl_vertices, jmask, vmask)
    """ Update Weights """
    loss = compute_loss(jv, target_jv) # squared l2 loss
    loss_std = np.sqrt(np.mean(loss))
    losses.append(loss_std)
    sigma2 = np.median(loss)
    weights = (np.ones(loss.shape)*sigma2)/(loss+np.ones(loss.shape)*sigma2)
    weighted_loss = loss*weights
    weighted_loss_std = np.sqrt(np.sum(weighted_loss)/np.sum(weights))
    print('\riter=%d, std=%f, weighted_std=%f, delta params=%f' % (i, loss_std, weighted_loss_std, np.linalg.norm(dparams, 2)), end="")
    #print('sigma=%f, max_weight=%f, min_weight=%f, median=%f' % (sigma, weights.max(), weights.min(), np.median(weights)))

  return params, losses, weights
