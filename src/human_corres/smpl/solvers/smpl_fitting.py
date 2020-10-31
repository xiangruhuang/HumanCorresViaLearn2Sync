import numpy as np

def smpl_fitting(points, label, model, max_iter=100, lamb=10.0):
  """
  Fitting an SMPL model to a point cloud, given correspondences.

  Parameter:
  ----------
  points: [N, 3]. Point Clouds.

  label: [N]. Corresponding vertices on the template for each point.

  model: SMPLModel initialized with shape and pose blend directions.

  max_iter: a integer. maximum number of iterations

  lamb: a float. Gauss-Newton second order parameters

  Return:
  -------
  params: vector of shape [85].
  losses: losses after each iteration.
  """
  params = np.zeros(85) # np.random.randn(85)
  p = points.reshape(-1)
  def update_vertices(model, params):
    model.set_params(trans=params[:3],
                     beta=params[3:13],
                     pose=params[13:])
    return model.verts

  vertices = update_vertices(model, params)
  grad = model.compute_derivatives()['v']
  losses = []
  for i in range(max_iter):
    # Jacobian matrix of shape [N*3, 85]
    J = model.compute_derivatives()['v'][label, :].reshape((-1, params.shape[0]))
    """
    x === params
    v === vec(vertices)
    p = vec(points)

    min_{dx} ||J dx + v - p||^2 + lambda || dx ||^2 gives
    Q dx = b
    Q = (J^T J + lambda I)
    b = J^T(p-v)

    """
    v = vertices[label, :].reshape(-1)
    Q = J.T.dot(J) + lamb*np.eye(params.shape[0])
    #import matplotlib.pyplot as plt
    #plt.imshow(Q)
    #plt.show()
    b = J.T.dot(p-v)
    dparams = np.linalg.solve(Q, b)
    params += dparams
    vertices = update_vertices(model, params)
    loss = np.square(np.linalg.norm(vertices[label, :]-points, 2, axis=1))
    loss_sum = np.sum(loss_sum) / len(loss)
    losses.append(loss_sum)
    print('iter=%d, loss=%f, delta params=%f' % (i, loss_sum, np.linalg.norm(dparams, 2)))

  return params, losses
