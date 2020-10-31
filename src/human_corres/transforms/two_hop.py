import torch
from torch_sparse import spspmm, coalesce

from torch_geometric.utils import remove_self_loops

class TwoHop(object):
  r"""Adds the two hop edges to the edge indices."""

  def __call__(self, edge_index, n):
    edge_index, _ = remove_self_loops(edge_index)
    fill = 1e16
    value = edge_index.new_full(
      (edge_index.size(1), ), fill, dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, n, n, n, coalesced=True)
    index, value = remove_self_loops(index, value)

    edge_index = torch.cat([edge_index, index], dim=1)
    edge_index, _ = coalesce(edge_index, None, n, n)

    return edge_index

  def __repr__(self):
    return '{}()'.format(self.__class__.__name__)
