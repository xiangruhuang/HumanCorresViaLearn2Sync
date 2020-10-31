from human_corres.config import PATH_TO_DATA
from geop.geometry.camera import PinholeCamera
from geop import linalg
from human_corres.utils import visualization as vis
from human_corres.utils import helper
import numpy as np
import scipy.io as sio
from sklearn.neighbors import NearestNeighbors as NN
import os
import argparse
import open3d as o3d
import glob

if __name__ == '__main__':
  SCAN = '%s/MPI-FAUST/training/scans/tr_scan_{0:03d}.ply' % (PATH_TO_DATA)
  SCAN_CORRES = '%s/MPI-FAUST/training/scans/tr_scan_{0:03d}.corres' % (PATH_TO_DATA)
  REG = '%s/MPI-FAUST/training/registrations/tr_reg_{0:03d}.ply' % (PATH_TO_DATA)
  FAUST_DESC = '%s/faust/faust_descriptors/{0:03d}.mat' % (PATH_TO_DATA)
  rotation_path = '%s/faust/render_rotations' % (PATH_TO_DATA)
  render_path = '%s/faust/scans' % (PATH_TO_DATA)
  OBJ = '%s/{0:03d}_{1:03d}.obj' % (render_path)
  CORRES = '%s/{0:03d}_{1:03d}.corres' % (render_path)
  MAT = '%s/{0:03d}_{1:03d}.mat' % (render_path)

  #camera = PinholeCamera()
  #rest_mesh = helper.loadSMPLModels()[0]
  #edges = computeGraph(6890, rest_mesh.faces, knn=7)
  #rest_mesh = vis.getTriangleMesh(rest_mesh.verts, rest_mesh.faces)
  gt_descriptors = helper.loadSMPLDescriptors('Laplacian_n')
  dsc_tree = NN(n_neighbors=1, n_jobs=10).fit(gt_descriptors)
  for scan_id in range(100):
    print(scan_id)
    """ Correspondence translation """
    raw_mesh = o3d.io.read_triangle_mesh(SCAN.format(scan_id)) # raw mesh
    reg_mesh = o3d.io.read_triangle_mesh(REG.format(scan_id)) # registration mesh
    tree = NN(n_neighbors=1, n_jobs=10).fit(np.array(reg_mesh.vertices))
    dists, indices = tree.kneighbors(np.array(raw_mesh.vertices))
    scan2reg = indices[:, 0]
    Nraw = np.array(raw_mesh.vertices).shape[0]
    Nreg = np.array(reg_mesh.vertices).shape[0]

    mesh = o3d.io.read_triangle_mesh(SCAN.format(scan_id))
    dsc_file = FAUST_DESC.format(scan_id)
    gt_dsc = sio.loadmat(dsc_file)['dsc'] # [6890, 128]
    _, gt_correspondence = dsc_tree.kneighbors(gt_dsc)
    gt_correspondence = gt_correspondence[:, 0]
    np.savetxt(SCAN_CORRES.format(scan_id), gt_correspondence, fmt='%d')
