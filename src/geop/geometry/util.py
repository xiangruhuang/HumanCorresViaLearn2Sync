import numpy as np

def cross_op(r):
  """
  Return the cross operator as a matrix
  i.e. for input vector r \in \R^3
  output rX s.t. rX.dot(v) = np.cross(r, v)
  where rX \in \R^{3 X 3}
  """
  rX = np.zeros((3, 3))
  rX[0, 1] = -r[2]
  rX[0, 2] = r[1]
  rX[1, 2] = -r[0]
  rX = rX - rX.T
  return rX

def rodrigues(r):
  """
  Return the rotation matrix R as a function of (axis, angle)
  following Rodrigues rotation theorem.
  (axis, angle) are represented by an input vector r, where
  axis = r / l2_norm(r) and angle = l2_norm(r)
  """
  theta = np.linalg.norm(r, 2)
  if theta < 1e-12:
    return np.eye(3)
  k = r / theta
  """ Rodrigues """
  R = np.cos(theta)*np.eye(3) + np.sin(theta)*cross_op(k) + (1-np.cos(theta))*np.outer(k, k)
  return R

def pack(R, t):
  """
    T = [[R, t]; [0, 1]]

  """
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = t.reshape(3)
  return T

def unpack(T):
  """ R = T[:3, :3]; t = T[:3, 3]
  """
  R = T[:3, :3]
  t = T[:3, 3]
  return R, t
