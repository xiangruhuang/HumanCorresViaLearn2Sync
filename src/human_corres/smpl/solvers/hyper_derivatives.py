import numpy as np

def compute_derivatives(model, correspondences, hyperparams):
  # initialization
  params = np.concatenate([model.trans, model.beta, model.pose], axis=0)
  weightNN = hyperparams[0]
  weightFeat = hyperparams[1]
  weightPoint2Plane = hyperparams[2]
  weightReg = hyperparams[3]
  lamb = hyperparams[4]
  indices2d = correspondences['indices2d'].astype(np.int32)
  indices3d = correspondences['indices3d'].astype(np.int32)
  joint_indices3d = correspondences['joint_indices3d'].astype(np.int32)
  n_hyper = hyperparams.shape[0]

  # initialize dictionaries
  derivative_dict = {}
  value_dict = {}
  derivative_dict['FuncVal'] = np.zeros(n_hyper)
  value_dict['FuncVal'] = 0.0

  # optimization variables
  Diag = np.eye(params.shape[0])
  Diag[3:13, 3:13] *= 10.0
  Q = (lamb+weightReg)*Diag
  b = -weightReg*params

  derivative_dict['FuncVal'][3] += 0.5*np.linalg.norm(params, 2)**2
  value_dict['FuncVal'] += 0.5*weightReg*np.linalg.norm(params, 2)**2

  # dQ, db
  derivative_dict['Q'] = np.zeros((85, 85, n_hyper))
  derivative_dict['b'] = np.zeros((85, n_hyper))
  derivative_dict['Q'][:, :, 3] = Diag # dQ/dweightReg
  derivative_dict['b'][:, 3] = -params # db/dweightReg
  derivative_dict['b'][:, :, 4] = Diag # dQ/dlamb

  derivatives = model.compute_derivatives()

