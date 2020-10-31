import numpy as np
import pickle

from geop import linalg
from geop.geometry import util as geo_util

class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    if model_path.endswith('.pkl'):
      with open(model_path, 'rb') as f:
        params = pickle.load(f, encoding='latin1')
    else:
      import scipy.io as sio
      params = sio.loadmat(model_path)
    """ J = #joints = 24
        N = #vertices = 6890
        F = #faces = 13776
    """
    self.J_regressor = params['J_regressor'] # [J, N]
    self.weights = params['weights'] # [N, J]
    self.posedirs = params['posedirs'] # [N, 3, (J-1)*9]
    self.v_template = params['v_template'] # [N, 3]
    self.shapedirs = params['shapedirs'] # [N, 3, 10]
    self.faces = params['f'] # [F, 3]
    self.kintree_table = params['kintree_table'] # [2, 24]

    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    } # number of keys = 24 
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    } # number of keys = 23

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None
    self.camera = (np.eye(3), np.eye(4))

    self.update()

  def set_params(self, pose=None, beta=None, trans=None):
    """
    Set pose, shape, and/or translation parameters of SMPL model. Verices of the
    model will be updated and returned.

    Parameters:
    ---------
    pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
    relative to parent joint. For root joint it's global orientation.
    Represented in a axis-angle format.

    beta: Parameter for model shape. A vector of shape [10]. Coefficients for
    PCA component. Only 10 components were released by MPI.

    trans: Global translation of shape [3].

    Return:
    ------
    Updated vertices.

    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if trans is not None:
      self.trans = trans
    self.update()
    return self.verts
  
  def update_params(self, params):
    self.set_params(trans=params[:3],
                    beta=params[3:13],
                    pose=params[13:])

  def set_camera(self, camera):
    self.camera = camera

  def compute_derivatives(self):
    """
    camera: tuple of (intrinsic, extrinsic). of shape ([3, 3,], [4, 4]).

    """
    import time
    self.derivatives = {}
    """ params = (trans, shape, pose) """
    N = self.v_template.shape[0]
    J = 24
    pose_shape=72
    # parameters are linearized
    # [N, 3, 3+10+72]
    self.derivatives['v_shaped'] = np.concatenate([
                                     np.zeros((N, 3, 3)),
                                     self.shapedirs,
                                     np.zeros((N, 3, pose_shape))
                                     ],
                                     axis=2
                                   )
    # [J, 3, D] = ([J, N]).dot([N, 3, D])
    self.derivatives['J0'] = np.tensordot(
                               self.J_regressor.toarray(),
                               self.derivatives['v_shaped'],
                               axes=[[1], [0]]
                             )
    pose_cube = self.pose.reshape((-1, 1, 3))
    drodriguez = [linalg.drodriguez(pose_cube[i, 0, :]) for i in range(J)]

    self.derivatives['R'] = np.zeros((J, 3, 3, 3+10+pose_shape))
    for i in range(J):
      self.derivatives['R'][i, :, :, (13+i*3):(13+(i+1)*3)] = drodriguez[i]

    self.derivatives['lrotmin'] = np.zeros((9*(J-1), 3+10+pose_shape))
    for i in range(1, J):
      self.derivatives['lrotmin'][(i-1)*9:i*9, (13+i*3):(13+(i+1)*3)] = drodriguez[i].reshape((9, 3))

    self.derivatives['v_posed'] = np.tensordot(
                                    self.posedirs,
                                    self.derivatives['lrotmin'],
                                    axes=[[2], [0]]
                                  ) + self.derivatives['v_shaped']
    self.derivatives['G_relative'] = np.zeros((J, 4, 4, 3+10+pose_shape))
    for i in range(J):
      self.derivatives['G_relative'][i, :3, :3, :] = self.derivatives['R'][i, :, :, :]
      self.derivatives['G_relative'][i, :3, 3, :] = self.derivatives['J0'][i, :, :]
      if i != 0:
        self.derivatives['G_relative'][i, :3, 3, :] -= self.derivatives['J0'][self.parent[i], :, :]
    # transpose to shape [J, 85, 4, 4] for computational convenience.
    dG_relative = np.transpose(self.derivatives['G_relative'], (0,3,1,2))
    dG = np.zeros(dG_relative.shape)
    G_relative = self.test_dict['G_relative'] # [J, 4, 4]
    G = np.zeros(G_relative.shape)
    G[0] = G_relative[0]
    dG[0] = dG_relative[0]
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(G_relative[i])
      dG[i, :, :, :] = dG[self.parent[i]].dot(G_relative[i])
      dG[i, :, :, :] += G[self.parent[i]].dot(dG_relative[i]).transpose((1,0,2))
    #print(G[4])
    #print(self.test_dict['G'])
    # transpose back to shape [J, 4, 4, 85]
    dG = np.transpose(dG, (0, 2, 3, 1))
    self.derivatives['G'] = dG
    A = G[:, :3, :3] # [J, 3, 3]
    dA = self.derivatives['G'][:, :3, :3, :].transpose((0,3,1,2)) # [J, 85, 3, 3]
    B = self.J0[:, np.newaxis, :, np.newaxis] # [J, 3, 1]
    dB = self.derivatives['J0'] # [J, 3, 85]
    AdB = np.matmul(A, dB) # [J, 3, 85]
    dAB = np.matmul(dA, B).squeeze(axis=3).transpose((0,2,1)) # [J, 3, 85]
    self.derivatives['offset'] = dAB + AdB

    self.derivatives['G_offset'] = np.array(self.derivatives['G'])
    self.derivatives['G_offset'][:, :3, 3, :] -= self.derivatives['offset']

    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    self.derivatives['T'] = np.tensordot(self.weights, self.derivatives['G_offset'], axes=[[1], [0]])
    # [N, 4, 4, 85]
    dT = self.derivatives['T'].transpose((0,3,1,2)) # [N, 85, 4, 4]
    v_posed = self.test_dict['v_posed'][:, np.newaxis, :, np.newaxis] # [N, 1, 3, 1]
    dv_posed = self.derivatives['v_posed'].transpose((0,2,1)) # [N, 85, 3]

    # [N, 85, 3, 1] <-- [N, 85, 3, 3] matmul [N, 1, 3, 1]
    dTv_posed = np.matmul(dT[:, :, :3, :3], v_posed)
    # [N, 85, 3, 1] <-- [N, 1, 3, 3] matmul [N, 85, 3, 1]
    Tdv_posed = np.matmul(T[:, np.newaxis, :3, :3], dv_posed[:, :, :, np.newaxis])

    # [3, 85]
    self.derivatives['trans'] = np.concatenate(
                                  [np.eye(3),
                                   np.zeros((3, 10)),
                                   np.zeros((3, 72))],
                                  axis=1
                                )
    # [N, 85, 3, 1] squeezes to [N, 85, 3] then transpose to [N, 3, 85]
    self.derivatives['v'] = (dTv_posed + Tdv_posed + dT[:, :, :3, 3:]).squeeze(axis=3).transpose((0, 2, 1))
    # [J, 3, 85] <-- [J, N] dot [N, 3, 85]
    self.derivatives['J'] = np.tensordot(self.J_regressor.toarray(),
                                         self.derivatives['v'],
                                         axes=[[1], [0]])

    # [N, 3, 85] += [1, 3, 85]
    self.derivatives['v'] += self.derivatives['trans'][np.newaxis, :, :]
    # [24, 3, 85] += [1, 3, 85]
    self.derivatives['J'] += self.derivatives['trans'][np.newaxis, :, :]
    ## [24, 3, 85]
    #R, trans = geo_util.unpack(self.camera[1])
    #J2d = self.camera[0].dot(R.dot(self.J.T)+trans[:, np.newaxis]).T # [24, 3]
    ## [3, 24, 85] = [3, 3] dot([[1], [1]]) [24, 3, 85]
    ##                   ^                       ^
    ##                   -----merging-------------
    #self.derivatives['J2d'] = np.tensordot(self.camera[0].dot(R),
    #                                       self.derivatives['J'],
    #                                       axes=[[1], [1]])
    ## [dx/dv, dy/dv, dz/dv]
    ## [d(x/z)/dv = x*d(1/z)/dv + dx/dv * (1/z),
    ##            = -x*z^{-2}*dz/dv + dx/dv * (1/z)]
    #J2d_x = J2d[:, 0][:, np.newaxis] # [24, 1]
    #J2d_y = J2d[:, 1][:, np.newaxis] # [24, 1]
    #inv_z = (1.0/J2d[:, 2])[:, np.newaxis] # [24, 1]
    ## [24, 85]
    #dJ2d_x = (-J2d_x*np.square(inv_z))*self.derivatives['J2d'][2]+self.derivatives['J2d'][0]*inv_z
    ## [24, 85]
    #dJ2d_y = (-J2d_y*np.square(inv_z))*self.derivatives['J2d'][2]+self.derivatives['J2d'][1]*inv_z
    #self.derivatives['J2d'] = np.stack([dJ2d_x, dJ2d_y], axis=1)

    ## [N, 24, 85] = [3, 3] dot([[1], [1]]) [N, 3, 85]
    ##                   ^                       ^
    ##                   -----merging-------------
    #self.derivatives['v2d'] = np.tensordot(self.camera[0].dot(R),
    #                                       self.derivatives['v'],
    #                                       axes=[[1], [1]])
    #v2d = self.camera[0].dot(R.dot(self.verts.T)+trans[:, np.newaxis]).T # [24, 3]
    ## [dx/dv, dy/dv, dz/dv]
    ## [d(x/z)/dv = x*d(1/z)/dv + dx/dv * (1/z),
    ##            = -x*z^{-2}*dz/dv + dx/dv * (1/z)]
    #v2d_x = v2d[:, 0][:, np.newaxis] # [24, 1]
    #v2d_y = v2d[:, 1][:, np.newaxis] # [24, 1]
    #inv_z = (1.0/v2d[:, 2])[:, np.newaxis] # [24, 1]
    ## [24, 85]
    #dv2d_x = (-v2d_x*np.square(inv_z))*self.derivatives['v2d'][2]+self.derivatives['v2d'][0]*inv_z
    ## [24, 85]
    #dv2d_y = (-v2d_y*np.square(inv_z))*self.derivatives['v2d'][2]+self.derivatives['v2d'][1]*inv_z
    #self.derivatives['v2d'] = np.stack([dv2d_x, dv2d_y], axis=1)
    return self.derivatives

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape.
    # [N, 3] <-- ([N, 3, 10]).dot([10])
    # [N, 3] <-- ([N, 3] + [N, 3])
    self.test_dict = {}
    self.test_dict['params'] = np.concatenate([self.trans, self.beta, self.pose.reshape(-1)])
    self.test_dict['v_template'] = self.v_template
    self.test_dict['shapedirs'] = self.shapedirs
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    self.test_dict['v_shaped'] = v_shaped
    # joints location
    # [J, 3] <-- ([J, N]).dot([N, 3])
    self.J0 = self.J_regressor.dot(v_shaped)
    self.test_dict['J0'] = self.J0
    # [J, 1, 3] <-- [J, 3].reshape((-1, 1, 3))
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    # [J, 3, 3] = rodrigues([J, 1, 3])
    self.R = self.rodrigues(pose_cube)
    self.test_dict['R'] = self.R
    # [J-1, 3, 3] = [np.eye(3), np.eye(3), np.eye(3), ...]
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    # [(J-1)* 9] = ([J-1, 3, 3] - [J-1,3,3]).flatten_by_row()
    lrotmin = (self.R[1:] - I_cube).ravel()
    self.test_dict['lrotmin'] = lrotmin
    # how pose affect body shape in zero pose
    # [N, 3] <-- [N, 3] + [N, 3, (J-1)*9].dot([(J-1)*9])
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    self.test_dict['v_posed'] = v_posed
    # world transformation of each joint
    # G = np.empty([J, 4, 4])
    G = np.empty((self.kintree_table.shape[1], 4, 4))
    #G[0] = self.__transformation__(self.R[0], self.J[0, :])
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J0[0, :].reshape([3, 1]))))
    for i in range(1, self.kintree_table.shape[1]):
      # [4, 4] <-- pack(R[i], J[i] - J[parent[i]])
      G[i] = self.__transformation__(
                   self.R[i], self.J0[i, :]-self.J0[self.parent[i], :]
                 )
                 #self.with_zeros(
                 #  np.hstack(
                 #    [self.R[i],((self.J0[i, :]-self.J0[self.parent[i],:]).reshape([3,1]))]
                 #  )
                 #)
    self.test_dict['G_relative'] = np.array(G)
    for i in range(1, self.kintree_table.shape[1]):
      G[i] = G[self.parent[i]].dot(G[i])
    self.test_dict['G'] = np.array(G)

    # remove the transformation due to the rest pose
    offset = self.pack(
      np.matmul(
        G,
        np.hstack([self.J0, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    offset = offset[:, :3, 3:4]
    # [J, 3] <-- [J, 3, 3] matmul [J, 3, 1]
    #offset = np.matmul(G[:, :3, :3], self.J[:, :, np.newaxis])
    self.test_dict['offset'] = offset.squeeze(axis=-1)
    G[:, :3, 3:4] -= offset    # transformation of each vertex
    self.test_dict['G_offset'] = G

    # [N, 4, 4] = np.tensordor([N, J], [J, 4, 4], axis=[1], [0])
    #   i.e. [i, j, k] = [i, :].dot([:, j, k])
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    self.test_dict['T'] = T
    # [N, 4] <-- ([N, 3], [N, 1])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    rest_shape_h = np.reshape(rest_shape_h, (-1, 4, 1))
    v = np.matmul(T, rest_shape_h).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])
    self.J = self.J_regressor.dot(v) + self.trans.reshape([1, 3]) # [24, 3]
    R, trans = geo_util.unpack(self.camera[1])
    self.J2d = self.camera[0].dot(R.dot(self.J.T)+trans[:, np.newaxis]).T # [24, 3]
    self.J2d = np.stack([self.J2d[:, 0]/self.J2d[:, 2],
                         self.J2d[:, 1]/self.J2d[:, 2]],
                        axis=1) # [24, 2]
    self.v2d = np.zeros((6890, 3, 85))
    #self.camera[0].dot(R.dot(self.verts.T)+trans[:, np.newaxis]).T # [24, 3]
    #self.v2d = np.stack([self.v2d[:, 0]/self.v2d[:, 2],
    #                     self.v2d[:, 1]/self.v2d[:, 2]],
    #                    axis=1) # [24, 2]
    self.test_dict['J'] = self.J
    self.test_dict['J2d'] = self.J2d
    self.test_dict['v2d'] = self.v2d
    self.test_dict['v'] = self.verts

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].

    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).tiny)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.concatenate([np.zeros((x.shape[0], 4, 3)), x], axis=2)
    #np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def __transformation__(self, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape((3))
    return T

  def save_to_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



if __name__ == '__main__':
  smpl = SMPLModel('model_intrinsics/male_model.mat')
  np.random.seed(9608)
  pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.0
  beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.0
  trans = np.zeros(smpl.trans_shape)
  smpl.set_params(beta=beta, pose=pose, trans=trans)
  smpl.save_to_obj('smpl_male.obj')
  import open3d as o3d
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(smpl.verts)
  mesh.triangles = o3d.utility.Vector3iVector(smpl.faces)
  mesh.compute_vertex_normals()

  o3d.io.write_triangle_mesh('smpl_male.ply', mesh)
