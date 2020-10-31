import torch
from torch.nn import Sequential as Seq, ReLU, BatchNorm1d as BN
from torch_geometric.nn import InstanceNorm

class LinearModule(torch.nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super(LinearModule, self).__init__(**kwargs)
    self.lin = torch.nn.Linear(in_channels, out_channels)
    self.norm = InstanceNorm(out_channels, affine=True)

  def forward(self, x_batch_tuple):
    x, batch = x_batch_tuple
    return (self.norm(ReLU()(self.lin(x)), batch), batch)

class MLPModule(torch.nn.Module):
  def __init__(self, channels, **kwargs):
    super(MLPModule, self).__init__(**kwargs)
    self.seq = Seq(*[
                 Seq(LinearModule(channels[i-1], channels[i]))
                 for i in range(1, len(channels))
               ])

  def forward(self, x, batch):
    output = self.seq((x, batch))
    return output[0]

if __name__ == '__main__':
  x = torch.randn(10, 3)
  batch = torch.zeros(10, dtype=torch.long)
  batch[:5] += 1
  A = MLPModule([3, 5, 7])
  x_out = A(x, batch)
  print(x_out.shape)
