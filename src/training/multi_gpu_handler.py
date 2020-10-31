import progressbar
import torch
import torch.nn.functional as F
from .utils import correspondence
from human_corres.smpl import smpl_align
import numpy as np
import scipy.io as sio

class MultiGPUHandler(object):
  def __init__(self, loader, args, lr_scheduler=None,
               training=True, reg=True):
    self.widgets = [progressbar.ETA()]
    if training:
      self.widgets.append(' training, ')
    else:
      self.widgets.append(' testing, ')
    self.widgets += [
      progressbar.Variable('loss'), ', ',
      progressbar.Variable('l2'), ', ',
      progressbar.Variable('recall'), ', ',
      progressbar.Variable('count'), ', ',
    ]
    if training:
      self.widgets.append(progressbar.Variable('lr'))
      self.lr_scheduler = lr_scheduler

    self.init = args.init
    if args.init:
      self.pool = { 'msg': [], 'errors': [] }
    self.template_points = torch.Tensor(loader.dataset.template_points).to(args.device)
    self.template_feats = torch.Tensor(loader.dataset.template_feats).to(args.device)
    self.total_loss = 0.0
    self.total_l2_loss_before_reg = torch.zeros(1)
    self.total_l2_loss_after_reg = torch.zeros(1)
    self.errors_before_reg = [0, 0, 0]
    self.errors_after_reg = [0, 0, 0]
    self.count = 0
    self.error_thresholds=[0.05, 0.1, 1e10]
    self.training, self.reg = (training, reg)

  def parse(self, out_dict, data_list, require_corres=False):
    if (len(data_list) > 0) and (data_list[0].y is not None):
      y = torch.cat([data.y for data in data_list]).to(out_dict['feats'].device)
      gt_feats = y[:, :-3]
      gt_points = y[:, -3:]
    else:
      y = None
    self.count += 1
    corres = correspondence(out_dict['feats'], self.template_feats)
    pred_points = self.template_points[corres, :]
    if require_corres:
      res_dict = {'ori_pos': [],
                  'pred_before_reg': [],
                  'pred_after_reg': []}
      corres_list = []
      offset = 0
      for i in range(len(data_list)):
        length = data_list[i].pos.shape[0]
        corres_i = corres[offset:(offset+length)]
        res_dict['ori_pos'].append(
          data_list[i].ori_pos.cpu().numpy()
        )
        res_dict['pred_before_reg'].append(
          pred_points[offset:(offset+length)].cpu().numpy()
        )
        if self.reg:
          res_dict['pred_after_reg'].append(
            out_dict['x_out0'][offset:(offset+length)].cpu().numpy()
          )
        offset += length
    if y is not None:
      errors = (pred_points - gt_points).norm(p=2, dim=-1)
      if self.init:
        self.pool['msg'].append(out_dict['msg'].detach().cpu().numpy())
        self.pool['errors'].append(errors.detach().cpu().numpy())
        return
      for i, error_threshold in enumerate(self.error_thresholds):
        labels_before_reg = errors < error_threshold
        self.errors_before_reg[i] += labels_before_reg.sum().item()
      self.total_l2_loss_before_reg[0] += errors.mean().item()
      if self.reg:
        loss = F.mse_loss(out_dict['x_out0'], gt_points)
        self.total_loss += loss.item()
        errors_after_reg = (out_dict['x_out0'] - gt_points).norm(p=2, dim=-1)
        for i, error_threshold in enumerate(self.error_thresholds):
          labels_after_reg = errors_after_reg < error_threshold
          self.errors_after_reg[i] += labels_after_reg.sum().item()
        self.total_l2_loss_after_reg[0] += errors_after_reg.mean().item()
      else:
        loss = F.mse_loss(out_dict['feats'], gt_feats)
        self.total_loss += loss.item()
    else:
      loss = None

    if require_corres:
      return loss, res_dict
    else:
      return loss

  def dump_msg(self, filename):
    msg = np.concatenate(self.pool['msg'], 0)
    errors = np.concatenate(self.pool['errors'], 0)
    sio.savemat(filename, {'msg': msg, 'errors': errors})

  def visualize(self, bar):
    l2_str = '({0:06f}, {1:06f})'.format(
                    (self.total_l2_loss_before_reg / self.count).item(),
                    (self.total_l2_loss_after_reg / self.count).item()
                    )
    recall_str = '(({0:04f}, {1:04f}), ({2:04f}, {3:04f}))'.format(
                    self.errors_before_reg[0]/max(self.errors_before_reg[-1], 1.0),
                    self.errors_before_reg[1]/max(self.errors_before_reg[-1], 1.0),
                    self.errors_after_reg[0]/max(self.errors_after_reg[-1], 1.0),
                    self.errors_after_reg[1]/max(self.errors_after_reg[-1], 1.0),
                    )
    if self.training:
      bar.update(
        self.count-1,
        loss=self.total_loss/self.count,
        l2=l2_str,
        recall=recall_str,
        count=self.count,
        lr=self.lr_scheduler.get_last_lr()[0],
      )
    else:
      bar.update(
        self.count-1,
        loss=self.total_loss/self.count,
        l2=l2_str,
        recall=recall_str,
        count=self.count,
      )


