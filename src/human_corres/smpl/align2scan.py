import scipy.io as sio
from human_corres.utils import helper, visualization as vis
from human_corres.smpl.solvers.solve import gauss_newton
from human_corres.smpl.solvers import smpl_nn_fitting as smpl_fitting
from human_corres.smpl.solvers.util import update_model
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN

def update_correspondences(fixed_points, fixed_normals,
                           moving_points, moving_normals,
                           correspondences, weights, sigma=1e-2):
  """ Compute Nearest Neighbor Distances

  """
  fixed = np.concatenate([fixed_points, 0.001*fixed_normals], axis=-1)
  moving = np.concatenate([moving_points, 0.001*moving_normals], axis=-1)
  tree = NN(n_neighbors=1).fit(moving)
  dists, indices = tree.kneighbors(fixed)
  dists = dists[:, 0]
  valid_idx = np.where(dists < 0.1)[0]
  dists = dists[valid_idx]
  indices = indices[valid_idx, 0].astype(np.int32)
  new_fixed_points = np.concatenate([fixed_points, fixed_points[valid_idx, :]], axis=0)
  new_fixed_normals = np.concatenate([fixed_normals, fixed_normals[valid_idx, :]], axis=0)
  new_correspondences = np.concatenate([correspondences, indices], axis=0)
  sigma2 = np.ones(dists.shape[0])*sigma*sigma
  new_weights = np.concatenate([weights, sigma2/(sigma2+np.square(dists))], axis=0)
  fixed_dict = {'points': new_fixed_points, 'normals': new_fixed_normals,
                'correspondences': new_correspondences,
                'weights': new_weights}

  return fixed_dict

def update_correspondence_weights(fixed_points, fixed_normals,
                                  moving_points, moving_normals,
                                  correspondences, sigma=1e-2):
  diff = moving_points[correspondences, :]-fixed_points
  diff_plane = np.multiply(diff, fixed_normals).sum(axis=1)
  sigma2 = np.ones(diff_plane.shape[0])*sigma*sigma
  weights = sigma2/(sigma2+np.square(diff_plane))

  return weights

def align(points, correspondences, normals, model, weights=None, weightPoint2Plane=0.9, max_iter=100):
  """ Align a SMPL model to a set of points.

  Args:
    points: np.ndarray of shape [N, 3]. Raw point cloud.
    correspondences: integer np.ndarray of shape [N].
                     j = correspondences[i] indicates a edge between
                     points[i] and model.verts[j]
    normals: np.ndarray of shape [N, 3]. normals.
    model: SMPLModel object.
    weights: [N] floats. Indicating confidence score of each correspondence.
    weightPoint2Plane: a float in [0, 1].
    max_iter: maximum number of iterations.
  Returns:
    params: SMPL parameters.
  """
  if weights is None:
    weights = np.ones(correspondences.shape[0])
  fixed_dict = {
                'points': points, 'normals': np.array(normals),
                'correspondences': np.array(correspondences).astype(np.int32),
                'weights': weights,
               }
  params = np.random.randn(85)*1e-3
  update_model(model, params)
  for itr in range(max_iter):
    mesh = vis.getTriangleMesh(model.verts, model.faces)
    mesh.compute_vertex_normals()
    model.compute_derivatives()

    moving_points = model.verts
    moving_normals = np.array(mesh.vertex_normals)
    if itr > 30:
      fixed_dict['weights'][:] = 0.0
      new_fixed_dict = update_correspondences(
                         fixed_dict['points'], fixed_dict['normals'],
                         moving_points, moving_normals,
                         fixed_dict['correspondences'], fixed_dict['weights'])
      cur_fixed_dict = new_fixed_dict
    else:
      cur_fixed_dict = fixed_dict
    cur_fixed_dict['weights'] = update_correspondence_weights(
                                  cur_fixed_dict['points'],
                                  cur_fixed_dict['normals'],
                                  moving_points, moving_normals,
                                  cur_fixed_dict['correspondences'],
                                  )
    res_dict = gauss_newton(moving_points, cur_fixed_dict['points'],
                 cur_fixed_dict['normals'], cur_fixed_dict['correspondences'],
                 cur_fixed_dict['weights'], model.derivatives['v'],
                 weightPoint2Plane)
    params += res_dict['dparams']
    update_model(model, params)
    #o3d.draw_geometries([vis.getTriangleMesh(model.verts, model.faces), vis.getPointCloud(points)])
    #print('dist=%4.6f, dparams=%3.4f' % (res_dict['dist'], np.linalg.norm(res_dict['dparams'], 2)))
  return params

def visualize_corres(points, correspondences, model):
  points += np.array([1.,0.,0.]).reshape((1, 3))
  tree = NN(n_neighbors=1000).fit(points)
  for idx in [88190, 132644]: #range(100):
    #idx = np.random.randint(points.shape[0]-1)
    print('visualizing around point %d' % idx)
    dists, indices = tree.kneighbors(points[idx, :].reshape((1, 3)))
    neighbors = indices[0, :]
    source = points[neighbors, :]
    #import ipdb; ipdb.set_trace()
    target = model.verts[correspondences[neighbors], :]
    edges = vis.getEdges(source, target, 1000, color=np.array([1.,0,0]))
    o3d.draw_geometries([vis.getPointCloud(points), vis.getTriangleMesh(model.verts, model.faces), edges])

def visualize_smoothness(points, correspondences, model):
  points += np.array([1.,0.,0.]).reshape((1, 3))
  tree = NN(n_neighbors=10).fit(points)
  dists, indices = tree.kneighbors(points)
  corres = correspondences[indices] # [n_points, 10]
  target_points = model.verts[corres, :] # [n_points, 10, 3]
  mean = target_points.mean(axis=1) # [n_points, 3]
  dists = target_points - mean[:, np.newaxis, :] # [n_points, 10, 3]
  dists = np.square(dists).sum(axis=1).sum(axis=1) # [n_points]
  max_dist = dists.max()
  min_dist = dists.min()
  dists = (dists - min_dist) / (max_dist - min_dist)
  r = np.array([1.0,0,0])
  b = np.array([0,0,1.])
  colors = np.array([(r*(di) + b*(1.0-di)) for di in dists])
  pcd = vis.getPointCloud(points)
  pcd.colors = o3d.utility.Vector3dVector(colors)
  o3d.draw_geometries([pcd, vis.getTriangleMesh(model.verts, model.faces)])

def smpl_align(points, correspondences, weights=None, max_iter=30):
  """
  Args:
    points: [N, 3] numpy array
    correspondences: [N]
  Returns:
    indices: [N] updated correspondence
  """
  N = points.shape[0]
  model = helper.loadSMPLModels()[0]
  if weights is None:
    weights = np.ones(N)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd.estimate_normals()
  normals = np.array(pcd.normals)
  params = align(points, correspondences, normals, model, weights,
                 max_iter=max_iter)
  model.update_params(params)
  tree = NN(n_neighbors=1, n_jobs=10).fit(model.verts)
  dists, indices = tree.kneighbors(points)
  indices = indices[:, 0]
  return indices

def smpl_align_mesh(vertices, faces, correspondences, weights=None, max_iter=30):
  """
  Args:
    points: [N, 3] numpy array
    correspondences: [N]
  Returns:
    indices: [N] updated correspondence
  """
  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.compute_vertex_normals()
  N = vertices.shape[0]
  model = helper.loadSMPLModels()[0]
  if weights is None:
    weights = np.ones(N)
  #pcd = o3d.geometry.PointCloud()
  #pcd.points = o3d.utility.Vector3dVector(points)
  #pcd.estimate_normals()
  #normals = np.array(pcd.normals)
  normals = np.array(mesh.vertex_normals)
  params = align(vertices, correspondences, normals, model, weights,
                 max_iter=max_iter)
  model.update_params(params)
  tree = NN(n_neighbors=1, n_jobs=10).fit(model.verts)
  dists, indices = tree.kneighbors(vertices)
  indices = indices[:, 0]
  return indices

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Align a Scan to a SMPL model',
             formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('--pointcloud', type=str, default=None,
    help="""A .ply file that contains a scan (point cloud).""")
  parser.add_argument('--mesh', type=str, default=None,
    help="""A .ply file that contains a mesh.""")
  parser.add_argument('--output', type=str, default=None,
    help="""A .ply file to output the result""")
  parser.add_argument('--correspondence', type=str,
                      default='examples/input.corres',
    help="""A text file that has the following format.
            1. First line contains text (without quotes):
               "point_id smpl_model_vertex_id_in_[0, 6890) confidence"
            2. for the rest lines, three numbers in each line
               S, T, c
               indicating a correspondence between
                 a) the S-th vertex from the geometry stored in args.ply_file
                 b) the T-th vertex in the standard SMPL model.
               with confidence score c (the larger the better).
         """)
  parser.add_argument('--max_iter', default=100, type=int,
    help='maximum number of gauss newton iterations')

  args = parser.parse_args()
  if (args.pointcloud is None) and (args.mesh is None):
    print('No Input Geometry Specified')
    args.print_help()
    exit(-1)
  if (args.pointcloud is not None) and (args.mesh is not None):
    print('Multiple Input Geometries Specified')
    args.print_help()
    exit(-1)

  if args.pointcloud is not None:
    geometry = o3d.io.read_point_cloud(args.pointcloud)
    points = np.array(geometry.points)
    o3d.estimate_normals(geometry, search_param=o3d.geometry.KDTreeSearchParamHybrid(
              radius=0.1, max_nn=30))
    normals = np.array(geometry.normals)
  else:
    geometry = o3d.io.read_triangle_mesh(args.mesh)
    geometry.compute_vertex_normals()
    points = np.array(geometry.vertices)
    normals = np.array(geometry.vertex_normals)

  model = helper.loadSMPLModels()[0]
  source_ids = []
  smpl_ids = []
  confidences = []
  with open(args.correspondence, 'r') as fin:
    for line in fin.readlines()[1:]:
      tokens = line.strip().split(' ')
      source_ids.append(int(tokens[0]))
      smpl_ids.append(int(tokens[1]))
      confidences.append(float(tokens[2]))
  source_ids = np.array(source_ids).astype(np.int32)
  smpl_ids = np.array(smpl_ids).astype(np.int32)
  confidences = np.array(confidences).astype(np.int32)

  correspondences = smpl_ids.astype(np.int32)
  print(correspondences.dtype)
  points = points[source_ids, :]
  normals = normals[source_ids, :]
  weights = confidences[source_ids]
  #visualize_smoothness(points, correspondences, model)
  params = align(points, correspondences, normals, model, weights,
                 max_iter=args.max_iter)
  update_model(model, params)
  tree1 = NN(n_neighbors=1, n_jobs=10).fit(model.verts)
  o3d.io.write_triangle_mesh(args.correspondence.replace('.corres', '.deform.ply'), vis.getTriangleMesh(model.verts, model.faces))
  o3d.io.write_triangle_mesh(args.correspondence.replace('.corres', '.raw.ply'), geometry)
  dists, indices = tree1.kneighbors(points)

  params[:3] += np.array([1.0,0,0])
  update_model(model, params)
  mesh = vis.getTriangleMesh(model.verts, model.faces)
  import matplotlib.pyplot as plt
  visualizer = o3d.visualization.Visualizer()
  visualizer.create_window('ply', 1024, 768, # width, height
                           50, 50, True)
  tree = NN(n_neighbors=1).fit(np.array(mesh.vertices)-np.array([1.,0,0]).reshape((1, 3)))
  dists, indices = tree.kneighbors(points)
  correspondence_smpl = indices[:, 0]
  output_corres_file = args.correspondence.replace('.corres', '.icp0.corres')
  print('saving to %s' % output_corres_file)
  with open(output_corres_file, 'w') as fout:
    fout.write('point_id smpl_model_vertex_id_in_[0, 6890) confidence \n')
    for i in range(correspondence_smpl.shape[0]):
      fout.write('%d %d %f\n' % (i, correspondence_smpl[i], 1.0))
  dists = dists[:, 0]
  max_dist = dists.max()
  min_dist = dists.min()
  avg_dist = dists.mean()
  dists = (dists - dists.min())/(dists.max()-dists.min())
  r = np.array([1.,0,0])
  b = np.array([0.,0,1.0])
  colors = np.stack([(r*di + (1.0-di)*b) for di in dists], axis=0)
  if args.pointcloud is not None:
    geometry.colors = o3d.utility.Vector3dVector(colors)
  else:
    geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
  visualizer.add_geometry(mesh)
  visualizer.add_geometry(geometry)
  #o3d.draw_geometries([mesh, geometry])
  img = visualizer.capture_screen_float_buffer(True)
  figures = plt.figure(figsize=(12, 8))
  plt.title('MAX(red)=%3.4fcm, AVG=%3.4fcm' % (max_dist*100, avg_dist*100), fontsize=20)
  plt.imshow(img)
  plt.axis('equal')
  plt.axis('off')
  plt.savefig(args.output)
