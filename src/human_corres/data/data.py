import torch_geometric

class Data(torch_geometric.data.Data):
  def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
               pos=None, norm=None, face=None, ori_pos=None,
               **kwargs):
    super(Data, self).__init__(x, edge_index, edge_attr, y, pos,
                               norm, face, **kwargs)
    self.ori_pos = ori_pos 
