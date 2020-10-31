from torch import nn
import torch
import numpy as np

Pool = nn.MaxPool2d

def batchnorm(x):
  return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
  def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
    super(Conv, self).__init__()
    self.inp_dim = inp_dim
    self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
    self.relu = None
    self.bn = None
    if relu:
      self.relu = nn.ReLU()
    if bn:
      self.bn = nn.BatchNorm2d(out_dim)

  def forward(self, x):
    assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
    x = self.conv(x)
    if self.bn is not None:
      x = self.bn(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

class Residual(nn.Module):
  def __init__(self, inp_dim, out_dim):
    super(Residual, self).__init__()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm2d(inp_dim)
    self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
    self.bn2 = nn.BatchNorm2d(int(out_dim/2))
    self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
    self.bn3 = nn.BatchNorm2d(int(out_dim/2))
    self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
    self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
    if inp_dim == out_dim:
      self.need_skip = False
    else:
      self.need_skip = True

  def forward(self, x):
    if self.need_skip:
      residual = self.skip_layer(x)
    else:
      residual = x
    out = self.bn1(x)
    out = self.relu(out)
    out = self.conv1(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)
    out += residual
    return out

class Hourglass(nn.Module):
  """
  Args:
    num_stacks: number of stacked hourglass networks
    io_dim: input and output dimension (#channels)
    bn: batch normalization
    increase: increase of #channels
  """
  def __init__(self, num_stacks, io_dim, bn=None, increase=0):
    super(Hourglass, self).__init__()
    f = io_dim
    nf = f + increase
    self.up1 = Residual(f, f)
    # Lower branch
    self.pool1 = Pool(2, 2)
    self.low1 = Residual(f, nf)
    self.n = num_stacks
    # Recursive hourglass
    if self.n > 1:
      self.low2 = Hourglass(num_stacks-1, nf, bn=bn)
    else:
      self.low2 = Residual(nf, nf)
    self.low3 = Residual(nf, f)
    self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, x):
    up1  = self.up1(x)
    pool1 = self.pool1(x)
    low1 = self.low1(pool1)
    low2 = self.low2(low1)
    low3 = self.low3(low2)
    up2  = self.up2(low3)
    return up1 + up2

class HgNet(nn.Module):
  def __init__(self, inp_dim, oup_dim, inter_dim=128, bn=False, increase=0, **kwargs):
    super(HgNet, self).__init__()
    self.pre = nn.Sequential(
      Conv(inp_dim, 64, 7, 2, bn=True, relu=True),
      Residual(64, 128),
      Pool(2, 2),
      Residual(128, 128),
      Residual(128, inter_dim)
    )
    self.hg = nn.Sequential(
      Hourglass(4, inter_dim, bn, increase),
    )

    self.feature = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='nearest'),
      Residual(inter_dim, inter_dim),
      nn.Upsample(scale_factor=2, mode='nearest'),
      Residual(inter_dim, inter_dim),
      Conv(inter_dim, inter_dim, 1, bn=True, relu=True)
    )
    self.out = Conv(inter_dim, oup_dim, 1, relu=False, bn=False)

  def forward(self, data):
    imgs = data.img
    imgs = self.pre(imgs)
    imgs = self.hg(imgs)
    imgs = self.feature(imgs)
    imgs = self.out(imgs)
    sizes = [imgs.shape[0], imgs.shape[2], imgs.shape[3]]

    imgs = imgs.permute((0, 2, 3, 1))
    imgs_flat = imgs.reshape((-1, imgs.shape[-1]))
    batched_indices = torch.cat((data.batch[:, None], data.indices), 1)
    flat_indices = torch.zeros(batched_indices.shape[0],
      dtype=torch.long).to(batched_indices.device)
    for i in range(3):
      flat_indices *= sizes[i]
      flat_indices += batched_indices[:, i]
    values = imgs_flat.index_select(0, flat_indices)
    return values
