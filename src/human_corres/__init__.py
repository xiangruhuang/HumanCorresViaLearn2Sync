try:
  import human_corres.data
  import human_corres.datasets
  import human_corres.modules
  import human_corres.transforms
  import human_corres.utils
  import human_corres.smpl
  from .config import *
except Exception as e:
  print('error init human_corres :', e)
