import warnings
from itertools import chain

import torch
import torch_geometric
from torch_geometric.data import Batch
from human_corres.data import ImgBatch, ImgData

class DataParallel(torch_geometric.nn.DataParallel):
  def __init__(self, module, device_ids=None, output_device=None):
    super(DataParallel, self).__init__(module, device_ids, output_device)
    self.src_device = torch.device("cuda:{}".format(self.device_ids[0]))

  def forward(self, data_list):
    """"""
    if len(data_list) == 0:
      warnings.warn('DataParallel received an empty data list, which '
              'may result in unexpected behaviour.')
      return None

    if not self.device_ids or len(self.device_ids) == 1:  # Fallback
      if isinstance(data_list[0], ImgData):
        data = ImgBatch.from_data_list(data_list).to(self.src_device)
      else:
        data = Batch.from_data_list(data_list).to(self.src_device)
      return self.module(data)

    for t in chain(self.module.parameters(), self.module.buffers()):
      if t.device != self.src_device:
        raise RuntimeError(
          ('Module must have its parameters and buffers on device '
           '{} but found one of them on device {}.').format(
             self.src_device, t.device))

    inputs = self.scatter(data_list, self.device_ids)
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, None)
    extra = {}
    for name in ['edge_idx01', 'edge_idx11', 'edge_idx00', 'edge_idx12']:
      if outputs[0].get(name, None) is not None:
        edge_indices01 = [d[name] for d in outputs]
        for d in outputs:
          d.pop(name)
        extra[name] = self.merge_graph(edge_indices01, self.output_device)
    out_dict = self.gather(outputs, self.output_device)
    out_dict.update(extra)

    return out_dict

  def merge_graph(self, edge_indices, target_device):
    num_points = [(e.max(dim=-1, keepdim=True)[0]+1).to(target_device) for e in edge_indices]
    for i in range(len(edge_indices)):
      if i == 0:
        offset = num_points[i]
      else:
        edge_indices[i] = edge_indices[i].to(target_device) + offset
        offset += num_points[i]
    return torch.cat(edge_indices, dim=-1)

  #def gather(outputs, target_device):

  def scatter(self, data_list, device_ids):
    num_devices = min(len(device_ids), len(data_list))

    count = torch.tensor([data.num_nodes for data in data_list])
    cumsum = count.cumsum(0)
    cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)
    device_id = num_devices * cumsum.to(torch.float) / cumsum[-1].item()
    device_id = (device_id[:-1] + device_id[1:]) / 2.0
    device_id = device_id.to(torch.long)  # round.
    split = device_id.bincount().cumsum(0)
    split = torch.cat([split.new_zeros(1), split], dim=0)
    split = torch.unique(split, sorted=True)
    split = split.tolist()

    res = []
    for i in range(len(split) - 1):
      if isinstance(data_list[0], ImgData):
        data_i = ImgBatch.from_data_list(data_list[split[i]:split[i + 1]]).to(
                  torch.device('cuda:{}'.format(device_ids[i])))
      else:
        data_i = Batch.from_data_list(data_list[split[i]:split[i + 1]]).to(
                  torch.device('cuda:{}'.format(device_ids[i])))
      res.append(data_i)
    return res
