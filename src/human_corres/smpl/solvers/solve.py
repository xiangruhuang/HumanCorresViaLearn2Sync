import numpy as np

def gauss_newton(moving_points, fixed_points, fixed_normals,
                 correspondences, weights, jacob, weightPoint2Plane):
  """ Updating deformation parameters.

  Args:
    moving_points: np.ndarray of shape [N, 3]. SMPL model points.
    fixed_points: np.ndarray of shape [M, 3]. Raw point cloud.
    fixed_normals: np.ndarray of shape [M, 3]. Estimated normals from
                   raw point cloud.
    correspondences: Integer np.ndarray of shape [M]. Correspondences from
                     point cloud to SMPL model.
                     j = correspondences[i] indicates a connection between
                     moving_points[j] to fixed_points[i].
    weights: np.ndarray of shape [M].
    jacob: np.ndarray of shape [N, 3, n_params]. Jacobian matrix from
           deformation parameters to SMPL model points (fixed)
    weightPoint2Plane: a float in [0, 1]. indicating balance between
                       point-to-point and point-to-plane distances in objective
                       function.

  Returns:
    delta_params: np.ndarray of shape [n_params]. Changes to
                  deformation parameters.
  """

  """ Aligning source and target.
      source = moving_points[correspondences, :]
      target = fixed_points
      target_normals = fixed_normals
      r = weightPoint2Plane
      Objective Function:
        (1.0-r)*point2point(source, target)
        + r*point2plane(source, target, target_normals)
      J = d(source)/d(params) = jacob[correspondences, :, :]
  """
  J = jacob[correspondences, :, :] # [M, 3, n_params]
  source = np.array(moving_points[correspondences, :]) # [M, 3]
  target = np.array(fixed_points) # [M, 3]
  target_normals = np.array(fixed_normals) # [M, 3]

  """ Point to Point Distances.
  vectorize source and target into source_vec, target_vec.
  diff3d_vec = source_vec - target_vec
  W = diag(weights).
  point2point(source, target) = 0.5*diff3d_vec^T W diff3d_vec
  J_vec = d(source_vec)/d(params) = d(diff3d_vec)/d(params)
  diff3d_vec ~= diff3d_vec0 + <J_vec, dparams>
  point2point
    ~= 0.5*(diff3d_vec0 + <J_vec, dparams>)^T W (diff3d_vec0 + <J_vec, dparams>)
    <=> 0.5*dparams^T Q3d dparams + b3d^T dparams
    => dparams = Q3d^-1 b3d
    Q3d = J_vec.T W J_vec
    b3d = J_vec.T W diff3d_vec0
  """
  diff3d_vec = (source-target).reshape(-1) # [M*3]
  loss3d = 0.5*np.square(diff3d_vec).sum()
  J3d_vec = J.reshape((-1, J.shape[-1])) # [M*3, n_params]
  W3d = np.kron(weights, np.ones(3)) # [M] -> [M*3]
  J3dTW = np.multiply(J3d_vec.T, W3d)
        # [n_params, M*3] = [M*3, n_params] * [M*3], fast
  Q3d = J3dTW.dot(J3d_vec) # <[n_params, M*3], [M*3, n_params]>
  b3d = J3dTW.dot(diff3d_vec) # <n_params, M*3], [M*3]>

  """ Point to Plane Distances """
  diff_plane = ((source-target)*target_normals).sum(axis=1) # [M]
  loss_plane = 0.5*np.square(diff_plane).sum()
  dist_plane = np.abs(diff_plane).mean()
  J3d_plane = np.multiply(J, target_normals[:, :, np.newaxis]).sum(axis=1)
        # [M, n_params] = ([M, 3, n_params] * [M, 3, 1]).sum(axis=1)
  J3d_planeTW = np.multiply(J3d_plane.T, weights)
        # [n_params, M] = [M, n_params] * [M], fast
  Q_plane = J3d_planeTW.dot(J3d_plane) # <[n_params, M], [M, n_params]>
  b_plane = J3d_planeTW.dot(diff_plane) # <n_params, M*3], [M*3]>

  """ Combination """
  #loss = loss3d*(1.0-weightPoint2Plane) + loss_plane*weightPoint2Plane
  loss = np.mean(np.abs(diff_plane))
  Q = Q3d*(1.0-weightPoint2Plane) + Q_plane*weightPoint2Plane
  b = b3d*(1.0-weightPoint2Plane) + b_plane*weightPoint2Plane

  """ Gauss Newton """
  Q += np.eye(Q.shape[0])*1.0
  dparams = np.linalg.solve(Q, -b)

  return {'dparams': dparams, 'loss': loss, 'dist': dist_plane}
