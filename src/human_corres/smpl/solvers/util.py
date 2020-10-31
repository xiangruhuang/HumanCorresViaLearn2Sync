def update_model(model, params):
  """Change model deformation parameters to ``params''.

  """
  model.set_params(trans=params[:3],
                   beta=params[3:13],
                   pose=params[13:])
