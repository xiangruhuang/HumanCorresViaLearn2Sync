import re

import torch
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected

class RadiusGraph(object):
  def __init__(self, radius):
    self.radius = radius

  def __call__(self, data):
    edge_index = radius_graph(data.pos, self.radius, None,
                              max_num_neighbors=64)
    edge_index = to_undirected(edge_index, num_nodes=data.pos.size(0))
    data.edge_index = edge_index
    return data

  def __repr__(self):
    return '{}(r={})'.format(self.__class__.__name__, self.radius)
