import scipy.io as sio
from torch_geometric.nn import knn_interpolate

class Prediction(object):
  def __init__(self, pos, x):
    self.pos = pos
    self.x = x

  def save_to_mat(self, mat_file):
    sio.savemat(mat_file, 
      {'pos': self.pos.cpu().numpy(),
       'x': self.x.cpu().numpy(),
       'errors': self.errors.cpu().numpy()
      })
  
  def evaluate_errors(self, y, threshold=0.1):
    self.errors = (self.x - y).norm(p=2, dim=-1)
    return self.errors
    #return errors.mean(), (errors < threshold).float().mean()

  def knn_interpolate(self, pos, k):
    x = knn_interpolate(self.x, self.pos, pos, k=k)
    return Prediction(pos, x)
