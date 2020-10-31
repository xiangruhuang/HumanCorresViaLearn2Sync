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
from human_corres.modules import RegularizationModule
from human_corres.modules import Regularization2Module
from human_corres.modules import Regularization3Module
from human_corres.modules import PointNet2
from human_corres.utils import helper

class HybridModel(torch.nn.Module):
  def __init__(self, args):
    super(HybridModel, self).__init__()
    self.feat_dim = args.embed_dim
    if args.cls:
      oup_dim = 6890
    else:
      oup_dim = self.feat_dim
    self.fe = PointNet2(inp_dim=0, oup_dim=self.feat_dim)
    if args.transf_reg:
      if args.RegNet == 'Reg2':
        self.reg = Regularization2Module()
      else:
        if args.RegNet == 'Reg':
          self.reg = RegularizationModule(init=args.init)
      if args.animals:
        gt_feats = torch.Tensor(helper.loadSMALDescriptors(args.desc)[:, :self.feat_dim])
        gt_points = torch.Tensor(np.array(helper.loadSMALModels()['cat'].vertices))
      else:
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
