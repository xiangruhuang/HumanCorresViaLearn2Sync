import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
#from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
#from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius
from torch_geometric.nn import knn_interpolate
from hybrid_corres.modules import Regularization2Module, Regularization3Module, RegularizationModule
from hybrid_corres.modules import PointNet2, PointNet2InstNorm
from hybrid_corres.utils import helper

class HybridModel(torch.nn.Module):
  def __init__(self, args):
    super(HybridModel, self).__init__()
    self.feat_dim = args.embed_dim
    if args.cls:
      oup_dim = 6890
    else:
      oup_dim = self.feat_dim
    #if args.FEnet == 'HgNet':
    #  self.fe = HgNet(
    #    inp_dim=5,
    #    oup_dim=self.feat_dim,
    #    bn=True,
    #  )
    if args.FEnet == 'PointNet2':
      self.fe = PointNet2(inp_dim=0, oup_dim=self.feat_dim)
    elif args.FEnet == 'PointNet2InstNorm':
      self.fe = PointNet2InstNorm(inp_dim=0, oup_dim=self.feat_dim)
    #elif args.FEnet == 'GraphSAGE':
    #  self.fe = GraphSAGE(3, self.feat_dim, oup_dim, args.cls)
    #elif args.FEnet == 'GAT':
    #  self.fe = GAT(3, self.feat_dim, oup_dim, args.cls)
    #elif args.FEnet == 'GCN':
    #  self.fe = GCN(3, self.feat_dim, oup_dim, args.cls)
    #elif args.FEnet == 'GraphUNet':
    #  self.fe = GraphUNet(3, self.feat_dim, oup_dim, args.cls)
    if args.transf_reg:
      if args.RegNet == 'Reg2':
        self.reg = Regularization2Module()
      elif args.RegNet == 'Reg3':
        self.reg = Regularization3Module()
      else:
        self.reg = RegularizationModule()
        #assert False, 'RegNet not implemented Yet!'
      gt_feats = torch.Tensor(helper.loadSMPLDescriptors(args.desc)[:, :self.feat_dim])
      gt_points = torch.Tensor(helper.loadSMPLModels()[0].verts)
      self.register_buffer('gt_feats', gt_feats)
      self.register_buffer('gt_points', gt_points)
    self.transf_reg = args.transf_reg

  def predict(self, source_feats, target_feats, target_points, data):
    N = source_feats.shape[0]
    dp = torch.matmul(source_feats, target_feats.transpose(0, 1))
    source_squares = (source_feats*source_feats).sum(dim=-1, keepdim=True)
    target_squares = (target_feats*target_feats).sum(dim=-1, keepdim=True)
    dists = source_squares - 2*dp + target_squares.view(1, -1)
    scores, indices = torch.topk(-dists, k=3)
    scores = scores.softmax(dim=-1)
    selected_points = torch.index_select(target_points, 0, indices.view(-1)).view(-1, 3, 3)
    predicted_points = (selected_points * scores.unsqueeze(-1)).sum(dim=-2)
    return predicted_points

  def forward(self, data):
    feats = self.fe(data)
    result = {}
    result['feats'] = feats
    levels = []
    if self.transf_reg:
      predicted_points = self.predict(feats, self.gt_feats, self.gt_points, data)
      result['pred_points'] = predicted_points
      res = self.reg(predicted_points, data.pos, data.batch, data.y)
      result.update(res)

    return result
