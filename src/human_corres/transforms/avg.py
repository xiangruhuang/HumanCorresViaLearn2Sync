import os.path as osp

from torch_scatter import scatter_add

class WeightedAvg(object):
  def __call__(self, x, weights, edge_index, direction='target2source'):
    """
    x: [N, D]
    weights: [N, 1]
    edge_index: target to source
    """
    N = x.shape[0]
    if direction == 'target2source':
      target_idx, source_idx = edge_index
    else: 
      source_idx, target_idx = edge_index
    edge_x = x.index_select(dim=0, index=source_idx)
    edge_weights = weights.index_select(dim=0, index=source_idx)
    new_x = scatter_add(
              edge_x*edge_weights,
              index=target_idx,
              dim=0,
              dim_size=N,
              )
    weights = scatter_add(
                edge_weights,
                index=target_idx,
                dim=0,
                dim_size=N,
                )
    new_x = new_x / weights
    return new_x

  def __repr__(self):
    return '{}()'.format(self.__class__.__name__)
