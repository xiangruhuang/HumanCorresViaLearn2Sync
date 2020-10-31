import progressbar
import torch
import torch.nn.functional as F
from .utils import correspondence
from human_corres.smpl import smpl_align

class Handler(object):
  def __init__(self, loader, args, lr_scheduler,
               training=True, reg=True):
    self.widgets = [progressbar.ETA()]
    self.lr_scheduler = lr_scheduler
    if training:
      self.widgets.append(' training, ')
    else:
      self.widgets.append(' testing, ')
    self.widgets += [
      progressbar.Variable('loss'), ', ',
      progressbar.Variable('l2'), ', ',
      progressbar.Variable('recall'), ', ',
    ]
    if training:
      self.widgets.append(progressbar.Variable('lr'))

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

  def parse(self, out_dict, data):
    gt_feats = data.y[:, :-3]
    gt_points = data.y[:, -3:]
    self.count += 1
    corres = correspondence(out_dict['feats'], self.template_feats)
    pred_points = self.template_points[corres, :]
    errors = (pred_points - gt_points).norm(p=2, dim=-1)
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
    return loss

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
        lr=self.lr_scheduler.get_last_lr()[0],
      )
    else:
      bar.update(
        self.count-1,
        loss=self.total_loss/self.count,
        l2=l2_str,
        recall=recall_str,
      )
