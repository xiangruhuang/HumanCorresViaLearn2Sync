import torch
from torch_geometric.nn import PointConv, fps, radius, knn_interpolate
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN

class SAModule(torch.nn.Module):
  def __init__(self, ratio, r, nn):
    super(SAModule, self).__init__()
    self.ratio = ratio
    self.r = r
    self.conv = PointConv(nn)

  def forward(self, x, pos, batch):
    idx = fps(pos, batch, ratio=self.ratio, random_start=False)
    row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
              max_num_neighbors=128)
    edge_index = torch.stack([col, row], dim=0)
    x = self.conv(x, (pos, pos[idx]), edge_index)
    pos, batch = pos[idx], batch[idx]
    return x, pos, batch

class FPModule(torch.nn.Module):
  def __init__(self, k, nn):
    super(FPModule, self).__init__()
    self.k = k
    self.nn = nn

  def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
    x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
    if x_skip is not None:
      x = torch.cat([x, x_skip], dim=1)
    x = self.nn(x)
    return x, pos_skip, batch_skip

def MLP(channels, batch_norm=True):
  return Seq(*[
           Seq(torch.nn.Linear(channels[i - 1], channels[i]),
               ReLU(), BN(channels[i], track_running_stats=False))
           for i in range(1, len(channels))
           ])

class PointNet2(torch.nn.Module):
  def __init__(self, inp_dim, oup_dim):
    super(PointNet2, self).__init__()
    self.sa1_module = SAModule(0.2, 0.05, MLP([inp_dim+3, 64, 64, 128]))
    self.sa2_module = SAModule(0.3, 0.1, MLP([128 + 3, 128, 128, 128]))
    self.sa3_module = SAModule(0.3, 0.2, MLP([128 + 3, 256, 256, 256]))
    self.sa4_module = SAModule(0.3, 0.4, MLP([256 + 3, 512, 512, 512]))
    self.sa5_module = SAModule(0.3, 0.8, MLP([512 + 3, 512, 512, 1024]))

    self.fp5_module = FPModule(1, MLP([1024 + 512, 512, 512]))
    self.fp4_module = FPModule(3, MLP([512 + 256, 256, 256]))
    self.fp3_module = FPModule(3, MLP([256 + 128, 256, 256]))
    self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
    self.fp1_module = FPModule(3, MLP([128+inp_dim, 128, 128, 128]))

    self.lin1 = torch.nn.Linear(128, 128)
    self.lin2 = torch.nn.Linear(128, 128)
    self.lin3 = torch.nn.Linear(128, oup_dim)

  def forward(self, data):
    sa0_out = (data.x, data.pos, data.batch)
    sa1_out = self.sa1_module(*sa0_out)
    sa2_out = self.sa2_module(*sa1_out)
    sa3_out = self.sa3_module(*sa2_out)
    sa4_out = self.sa4_module(*sa3_out)
    sa5_out = self.sa5_module(*sa4_out)

    fp5_out = self.fp5_module(*sa5_out, *sa4_out)
    fp4_out = self.fp4_module(*fp5_out, *sa3_out)
    fp3_out = self.fp3_module(*fp4_out, *sa2_out)
    fp2_out = self.fp2_module(*fp3_out, *sa1_out)
    fp1_out = self.fp1_module(*fp2_out, *sa0_out)

    x = fp1_out[0]

    x = self.lin1(x)
    x = torch.nn.ReLU()(x)
    x = self.lin2(x)
    x = self.lin3(x)
    return x

