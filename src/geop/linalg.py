import numpy as np

""" Projection, from depth image to 2D pixels
Input:
    pointcloud: np.ndarray of shape [N, 3].
    intrinsic: camera intrinsic
    extrinsic: extrinsic parameters
Output:
    pixels: np.ndarray of shape [N, 2].
"""
def pointcloud2pixel(pointcloud, extrinsic, intrinsic):
  R = extrinsic[:3, :3]
  trans = extrinsic[:3, 3]
  pixels = intrinsic.dot(R.dot(pointcloud.T)+trans[:, np.newaxis]).T
  flip = np.array([[0,1,0],
                   [1,0,0],
                   [0,0,1]])
  pixels = np.linalg.inv(flip).dot(pixels.T).T
  pixels[:, 0] /= pixels[:, 2]
  pixels[:, 1] /= pixels[:, 2]
  return pixels[:, :2]

def depth2pointcloud(depth, ext, intc):
  """Inverse projection, from depth image to 3D point cloud.

  Args:
    depth: np.ndarray of shape [W, H] (0 indicates invalid depth).
    ext: extrinsic matrix in [4, 4]
    intc: intrinsic matrix in [3, 3]

  Returns:
    points: np.ndarray of shape [N, 3].
  """
  intrinsic = intc
  extrinsic = ext
  R = extrinsic[:3, :3]
  trans = extrinsic[:3, 3]
  x, y = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
  valid_idx = np.where(depth > 1e-7)
  z = depth[valid_idx]
  x = x[valid_idx]*z
  y = y[valid_idx]*z
  flip = np.array([[0,1,0],
                   [1,0,0],
                   [0,0,1]])
  points = flip.dot(np.stack([x, y, z], axis=1).T).T

  points = np.linalg.pinv(intrinsic).dot(points.T).T

  points = R.T.dot(points.T-trans[:, np.newaxis]).T

  return points

def cross_op(r):
  """
  matrix operator of cross product

  Parameter:
  ----------
  r: np.ndarray of shape [3]. A 3d vector.

  Return:
  -------
  A: a matrix s.t. Av is equivalent to np.cross(r, v)
  """
  x = r[0]
  y = r[1]
  z = r[2]
  A = np.array([[0,  -z,  y],
                 [z,   0, -x],
                 [-y,  x,  0]])
  return A

def dcross_op():
  """
  derivative of cross product operator w.r.t. the 3d vector

  Return:
  D: a tensor of shape [3, 3, 3]
     e.g. D[:, :, i] is the d cross_op(r) / r(i), for i = 0, 1, 2.
  """
  d1 = np.zeros((3, 3))
  d2 = np.zeros((3, 3))
  d3 = np.zeros((3, 3))
  d1[1, 2] = -1
  d1[2, 1] = 1
  d2[0, 2] = 1
  d2[2, 0] = -1
  d3[0, 1] =-1
  d3[1, 0] = 1
  return np.stack([d1, d2, d3], axis=2)

def douter(v):
  """
  derivative of np.outer(v, v) w.r.t. v

  Parameter:
  ----------
  v: a 3d vector.

  Return:
  -------
  D: a tensor of shape [3, 3, 3].
     e.g. D[:, :, i] is the d outer(r) / r(i), for i = 0, 1, 2.
  """
  e1 = np.array([1.0, 0, 0])
  e2 = np.array([0, 1.0, 0])
  e3 = np.array([0, 0, 1.0])
  d1 = np.outer(e1, v) + np.outer(v, e1)
  d2 = np.outer(e2, v) + np.outer(v, e2)
  d3 = np.outer(e3, v) + np.outer(v, e3)
  return np.stack([d1, d2, d3], axis=2)

def rot2axis_angle(R):
  """ Convert a rotation back to rotation axis and rotation angle.

  Args:
    R: np.ndarray of shape [3, 3]

  Returns:
    r: a np.ndarray of shape [3]. rotation axis * rotation angle
  """
  costheta = np.clip((np.diag(R).sum()-1.0)/2.0, -1, 1)
  theta = np.arccos(costheta)
  if theta < 1e-12:
    return np.zeros(3)
  x = R[2, 1] - R[1, 2]
  y = R[0, 2] - R[2, 0]
  z = R[1, 0] - R[0, 1]
  r = np.array([x, y, z])
  r = r / np.linalg.norm(r, 2) * theta
  assert (not np.isnan(r).any())

  return r

def rodriguez(r):
  """
  Compute rodrigues rotation operator

  Paramter:
  ---------
  r: a 3d vector. theta = np.linalg.norm(r, 2) represent the angle.
     k=r/theta represent the rotation axis.

  Return:
  -------
  R: a 3D rotation matrix computed from the theorem.
  """
  theta = np.linalg.norm(r, 2)
  if theta < 1e-12:
    return np.eye(3)
  else:
    k = r / theta
  sint = np.sin(theta)
  cost = np.cos(theta)
  return cost*np.eye(3) + sint*cross_op(k) + (1-cost)*np.outer(k, k)

def rotation_angle_error(R1, R2):
  RR = R1.dot(R2.T)
  costheta = np.clip((np.diag(RR).sum()-1.0)/2.0, -1, 1)
  theta = np.arccos(costheta)
  return theta

def drodriguez(r):
  """
  Compute derivative of rodrigues rotation operator

  Paramter:
  ---------
  r: a 3d vector. theta = np.linalg.norm(r, 2) represent the angle.
     k=r/theta represent the rotation axis.

  Return:
  -------
  A: a matrix of shape [3, 3, 3].
     A[:, :, i] = d rodriguez(r) / dr(i)
  """

  theta = np.linalg.norm(r, 2)
  if theta < 1e-15:
    return np.random.randn(3, 3, 3)*1e-1
  else:
    k = r / theta
  sint = np.sin(theta)
  cost = np.cos(theta)

  """ w.r.t. theta and k """
  dR_dtheta = -sint*np.eye(3) + cost*cross_op(k) + sint*np.outer(k, k)
  dR_dtheta = np.expand_dims(dR_dtheta, -1)

  """ Find unit vector k1, k2 perpendicular to k """
  k1 = np.cross(k, np.random.randn(3))
  while np.linalg.norm(k1, 2) < 1e-6:
    k1 = np.cross(k, np.random.randn(3))
  k1 = k1 / np.linalg.norm(k1, 2)
  k2 = np.cross(k, k1)
  k2 = k2 / np.linalg.norm(k2, 2)

  """ Let dk = k1*x + k2*y """
  douter_dxy = np.stack(
                 [douter(k).dot(k1), douter(k).dot(k2)],
                 axis=2
               )
  dcrossop_dxy = np.stack(
                   [dcross_op().dot(k1), dcross_op().dot(k2)],
                   axis=2
                 )
  dR_dxy = sint*dcrossop_dxy+(1-cost)*douter_dxy
  Rot = lambda inp: rodriguez(inp[0]*inp[1])
  #print((Rot((theta, k+k1*1e-7))-Rot((theta, k))) / 1e-7)
  #print(dR_dxy[:, :, 0])
  #import ipdb; ipdb.set_trace()

  """ w.r.t. r """
  dtheta_dr = r / theta
  dtheta_dr = np.expand_dims(dtheta_dr, 0)
  dk_dr = np.eye(3)*(1.0/theta) - np.outer(r, r)/(theta**3)
  dxy_dr = np.concatenate(
             [k1[np.newaxis, :].dot(dk_dr),
              k2[np.newaxis, :].dot(dk_dr)],
             axis=0
           )

  """ Connect """
  dR_dr1 = np.tensordot(dR_dtheta, dtheta_dr, axes=[[2], [0]])
  dR_dr2 = np.tensordot(dR_dxy, dxy_dr, axes=[[2], [0]])
  return dR_dr1+dR_dr2

if __name__ == '__main__':
  """ Testing cross operator """
  a = np.random.randn(3)
  b = np.random.randn(3)
  assert np.linalg.norm(np.cross(a, b)-cross_op(a).dot(b), 2) < 1e-8
  eps = 1e-6
  delta_a = np.random.randn(3)*eps
  assert np.linalg.norm((cross_op(a+delta_a) - cross_op(a)) - dcross_op().dot(delta_a), 2) < 1e-10

  """ Testing rodrigues """
  #drodriguez = drodriguez(a)
  #import ipdb; ipdb.set_trace()
  #axis = lambda a: a / np.linalg.norm(a, 2)
  #angle = lambda a: np.linalg.norm(a, 2)
  #assert np.linalg.norm(dk_da.dot(delta_a) - (axis(a+delta_a) - axis(a))) < 1e-10
  #assert np.linalg.norm(dtheta_da.dot(delta_a) - (angle(a+delta_a) - angle(a))) < 1e-10
  #np.linalg.norm(dtheta_dr)
  assert np.linalg.norm(rodriguez(a+delta_a)-rodriguez(a) - drodriguez(a).dot(delta_a), 'fro') < 1e-10
