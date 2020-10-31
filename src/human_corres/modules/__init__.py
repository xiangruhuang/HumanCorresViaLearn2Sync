try:
  from .pointnet2 import PointNet2
  #from .pointnet2_instnorm import PointNet2InstNorm
  #from .graph_sage import GraphSAGE
  #from .gat import GAT
  #from .gcn import GCN
  #from .graph_unet import GraphUNet
  #from .diffpool import DiffPool
  #from .hourglass import HgNet
  #from .transformation import TransformationModule
  #from .regularization import RegularizationModule, ConfEstModule
  #from .conf_est import ConfEstModule
  from .reg import Regularization2Module, Regularization3Module, RegularizationModule
  from .pipeline import HybridModel
except Exception as e:
  print('error init human_corres.modules:', e)
