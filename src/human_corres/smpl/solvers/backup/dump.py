


  #""" 2D correspondences """
  #if (len(indices2d) > 0) and False:
  #  source3d_pool = np.concatenate([model.verts, model.J], axis=0)
  #  source2d = projection2d(model, source3d_pool)[indices2d, :]
  #  target2d = correspondences['target2d']
  #  J2d = np.concatenate([derivatives['v2d'], derivatives['J2d']], axis=0)
  #  J2d = J2d[indices2d, :, :].reshape((-1, params.shape[0]))
  #  J2d[:, 3:13] = 0.0
  #  W2d = np.diag(np.kron(correspondences['weights2d'], np.ones(2)))
  #  Q2d = J2d.T.dot(W2d).dot(J2d)
  #  diff2d = target2d-source2d
  #  b2d = J2d.T.dot(W2d).dot(diff2d.reshape(-1))
  #  loss2d = np.mean(np.square(np.linalg.norm(diff2d, 2, axis=-1)))
  #  if stats is not None:
  #    stats['loss2d'].append(loss2d)
  #  dparams_2d = np.linalg.pinv(Q2d).dot(b2d) # np.linalg.solve(Q2d, b2d)
  #  #print('dparams2d=%f' % np.linalg.norm(dparams_2d, 2))
  #  Q += Q2d*0.01
  #  b += b2d*0.01
  #else:
  #  if stats is not None:
  #    stats['loss2d'].append(np.nan)
