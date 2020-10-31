try:
  from .rigid import rigid_fitting, hierarchical_rigid_fitting
except Exception as e:
  print('error init human_corres.utils :', e)
