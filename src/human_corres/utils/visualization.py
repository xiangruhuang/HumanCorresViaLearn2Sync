import numpy as np
import open3d as o3d
import sys, os
project_path = os.path.abspath(__file__)
project_path = os.path.dirname(project_path)
sys.path.append(project_path)
from sklearn.neighbors import NearestNeighbors as NN
import helper

def color(ratio):
  return ratio*np.array([1.0,0,0]) + (1.0-ratio)*np.array([0,1.0,0.0])

def paint_cls_result(pcd, labels, color_pos, color_neg):
  points = np.array(pcd.points)
  colors = np.zeros((points.shape[0], 3))
  colors[labels] = np.array(color_pos).reshape(1, 3)
  colors[labels == False] = np.array(color_neg).reshape(1, 3)
  pcd.colors = o3d.utility.Vector3dVector(colors)
  return pcd

def visualize_errors(points_list, errors_list, threshold=0.05, translate=True):
  pcds = []
  for i, points_errors in enumerate(zip(points_list, errors_list)):
    points, errors = points_errors
    if translate:
      points = points + 0.0
      points[:, 0] += i*1.0
    N = points.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((N, 3))
    colors[errors < threshold, 2] = 1.0
    colors[errors > threshold, 0] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcds.append(pcd)
  o3d.visualization.draw_geometries(pcds)

def visualize_error(points, errors, threshold=0.05):
  N = points.shape[0]
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  colors = np.zeros((N, 3))
  colors[errors < threshold, 2] = 1.0
  colors[errors > threshold, 0] = 1.0
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.visualization.draw_geometries([pcd])

def visualize_points(points_list):
  """Visualize list of Points with different colors.

  Args:
    points_list: list of [n_i, 3] np.ndarray.
  """

  n = len(points_list)
  pcds = []
  for i, points in enumerate(points_list):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color(i*1.0/n))
    pcds.append(pcd)
  o3d.visualization.draw_geometries(pcds)

def visualize_correspondence_smoothness(points, correspondence, max_error, min_error):
  """ Visualize smoothness of predicted correspondences
  Args:
    points: o3d.geometry.TriangleMesh, contains N points.
    correspondence: [N]
  """
  pcd = getPointCloud(points)
  model = helper.loadSMPLModels()[0]
  pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)+np.array([1.0,0,0]).reshape((1, 3)))
  v = np.array(pcd.points)
  N = v.shape[0]
  tree = NN(n_neighbors=20).fit(v)
  dists, indices = tree.kneighbors(v)
  target = model.verts[correspondence, :] # [N, 3]
  centers = target[indices, :].mean(axis=1) # [N, 3]
  diff_norms = np.square(target[indices, :] - centers[:, np.newaxis, :]).sum(axis=-1).mean(axis=1) # [N]
  diff_norms[np.where(diff_norms > max_error)] = max_error
  diff_norms[np.where(diff_norms < min_error)] = min_error
  import ipdb; ipdb.set_trace()
  max_indices = np.argsort(diff_norms)[-1000:]
  edges = getEdges(v[max_indices, :], target[max_indices, :])
  colors = (diff_norms-min_error)/(max_error - min_error)
  r = np.outer(np.ones(N), np.array([1.0, 0, 0])) # [N, 3]
  b = np.outer(np.ones(N), np.array([0., 0, 1.0])) # [N, 3]
  colors = b*(1.0-colors[:, np.newaxis])+r*(colors[:, np.newaxis])
  pcd.points = o3d.utility.Vector3dVector(colors)
  o3d.draw_geometries([pcd, getTriangleMesh(model.verts, model.faces), edges])

def visualize_correspondences(raw, model, corres_dict, key='all'):
  """

  Args:
    source: [n, 3]
    target: SMPLModel
    corres_dict: dictionary
  """
  if key == 'joint':
    raw = np.concatenate([corres_dict['joint_target3d'], raw], axis=0)
    target = np.concatenate([model.J, model.verts], axis=0)
    corres = corres_dict['joint_indices3d'].astype(np.int32)
    weights = corres_dict['joint_weights3d']
  else:
    raw = np.concatenate([corres_dict['target3d'], raw], axis=0)
    target = np.concatenate([model.verts], axis=0)
    corres = corres_dict['indices3d'].astype(np.int32)
    weights = corres_dict['weights3d']

  pcd1 = o3d.geometry.PointCloud()
  pcd2 = o3d.geometry.PointCloud()
  pcd1.points = o3d.utility.Vector3dVector(raw)
  pcd2.points = o3d.utility.Vector3dVector(target)
  pcd1.paint_uniform_color(color(0))
  pcd2.paint_uniform_color(color(1))

  lines = []
  for i, ci in enumerate(corres[:100]):
    s = raw[i, :]
    t = target[ci, :]
    if weights[i] < 1e-6:
      continue
    for r in range(100):
      ratio = r / 100.0
      mid = s*ratio + t*(1-ratio)
      lines.append(mid)
  lines = np.array(lines)
  print(lines.shape)
  pcd3 = o3d.geometry.PointCloud()
  lines = lines.reshape((-1, 3))
  pcd3.points = o3d.utility.Vector3dVector(lines)
  pcd3.paint_uniform_color(color(0.5))
  o3d.draw_geometries([pcd1, pcd2, pcd3])

def getPointCloud(points):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  return pcd

def getTriangleMesh(vertices, triangles):
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  mesh.compute_vertex_normals()
  return mesh

def getEdges(sources, targets, num_samples=100, color=np.array([1.0, 0, 0])):
  if num_samples == -1:
    num_samples = sources.shape[0]
  indices = np.random.permutation(sources.shape[0])[:num_samples]
  lines = []
  for i in indices:
    s = sources[i, :]
    t = targets[i, :]
    ns = 100
    for r in range(ns):
      ratio = r / (ns-1.0)
      mid = s*ratio + t*(1-ratio)
      lines.append(mid)
  lines = np.array(lines)
  pcd = getPointCloud(lines)
  if color is not None:
    pcd.paint_uniform_color(color)
  return pcd

def visualize_embedding(raw, model, corres):
  """Visualize point-wise embedding of raw point clouds.

  """
  mesh = getTriangleMesh(model.verts, model.faces)
  raw += np.array([1.0, 0.0, 0.0]).reshape((1, 3))
  pcd = getPointCloud(raw)
  sources = raw
  targets = model.verts[corres, :]
  edges = getEdges(sources, targets)
  o3d.draw_geometries([mesh, pcd, edges])

def visualize_embedding_pcds(raw1, raw2, corres):
  """Visualize point-wise embedding of raw point clouds.
    
  """
  pcd1 = getPointCloud(raw1)
  #mesh = getTriangleMesh(model.verts, model.faces)
  raw2 += np.array([1.0, 0.0, 0.0]).reshape((1, 3))
  pcd2 = getPointCloud(raw2)
  sources = raw1
  targets = raw2 #model.verts[corres, :]
  edges = getEdges(sources, targets)
  o3d.draw_geometries([pcd1, pcd2, edges])

def visualize_fitting(model, raw):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(raw)
  pcd.paint_uniform_color(color(0))
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(model.verts)
  mesh.triangles = o3d.utility.Vector3iVector(model.faces)

  o3d.draw_geometries([mesh, pcd])
