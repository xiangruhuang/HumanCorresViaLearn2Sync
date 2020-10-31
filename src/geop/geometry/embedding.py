import numpy as np
import open3d as o3d
import os, os.path as osp

def laplacian(mesh):
  """Returns the laplacian matrix of this mesh.

  """
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  A = np.zeros((N, N))
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      A[(faces[:, i], faces[:, j])] = 1.0
  A = A + A.T
  diag = A.dot(np.ones(N))
  L = np.diag(diag) - A
  return L

def laplacian_embedding(mesh, rank=30):
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  A = np.zeros((N, N))
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      A[(faces[:, i], faces[:, j])] = 1.0
  A = A + A.T
  diag = A.dot(np.ones(N))
  L = np.diag(diag) - A
  eigvals, eigvecs = np.linalg.eigh(L)
  embedding = eigvecs[:, 1:(rank+1)]
  return embedding

def floyd(mesh):
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  Dist = np.zeros((N, N)) + 1e10
  for i in range(N):
    Dist[i, i] = 0.0
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      Dist[(faces[:, i], faces[:, j])] = 1.0
  #for k in range(N):
  #  print(k, N)
  #  for i in range(N):
  #    for j in range(N):
  #      if (i == j) or (i == k) or (j == k):
  #        continue
  #      if Dist[i, j] > Dist[i, k] + Dist[k, j]:
  #        Dist[i, j] = Dist[i, k] + Dist[k, j]
  return Dist

if __name__ == '__main__':
  #import scipy.io as sio
  #mesh = o3d.io.read_triangle_mesh('example_data/mesh_female.ply')
  #embedding = laplacian_embedding(mesh, rank=128)
  #embedding = (embedding - embedding.min(0)[np.newaxis, :])/(embedding.max(0)-embedding.min(0))[np.newaxis, :]
  #import ipdb; ipdb.set_trace()
  #sio.savemat('smpl_laplacian_embedding_128.mat', {'male': embedding, 'female': embedding})
  
  import scipy.io as sio
  mesh = o3d.geometry.TriangleMesh()
  import argparse
  parser = argparse.ArgumentParser('Compute Laplacian embedding of a given mesh (.ply)')
  parser.add_argument('--mesh', default=None, type=str,
    help='input mesh (.ply)')
  parser.add_argument('--mat', default=None, type=str,
    help='output dictionary (.mat)')
  parser.add_argument('--name', default=None, type=str,
    help='name')
  args = parser.parse_args()
 
  mesh = o3d.io.read_triangle_mesh(args.mesh)
  #mat = sio.loadmat('../../../data/TOSCA/horse0.mat')
  #import ipdb; ipdb.set_trace() 
  #triangles = np.array(mat['surface'][0,0]['TRIV']).astype(np.int32) - 1
  #x = np.array(mat['surface'][0,0]['X']).astype(np.float32)
  #y = np.array(mat['surface'][0,0]['Y']).astype(np.float32)
  #z = np.array(mat['surface'][0,0]['Z']).astype(np.float32)
  #mesh = o3d.geometry.TriangleMesh()
  #mesh.vertices = o3d.utility.Vector3dVector(np.concatenate([x, y, z], axis=-1)) 
  #mesh.triangles = o3d.utility.Vector3iVector(triangles) 
 
  embedding = laplacian_embedding(mesh, rank=128)
  embedding = (embedding - embedding.min(0)[np.newaxis, :])/(embedding.max(0)-embedding.min(0))[np.newaxis, :]
  if osp.exists(args.mat):
    append = True
  else:
    append = False
  sio.savemat(args.mat, { args.name: embedding }, appendmat=append)
  
  #N = np.array(mesh.vertices).shape[0]
  #Dist = np.loadtxt('floyd_results').reshape((6890, 6890))
  #dists = []
  #dists = Dist.reshape(-1)
  #feature_dists = np.linalg.norm(embedding[np.newaxis, :, :] - embedding[:, np.newaxis, :], 2, axis=-1).reshape(-1)
  #import matplotlib.pyplot as plt
  #feature_dists = np.array(feature_dists)
  #dists = np.array(dists)
  #random_idx = np.random.permutation(N*N)[:100000]
  #plt.scatter(dists[random_idx], feature_dists[random_idx], 1.0)
  #plt.savefig('hey2.png')
  #plt.show()
