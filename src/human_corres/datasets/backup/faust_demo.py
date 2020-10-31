import open3d as o3d
from hybrid_corres.config import PATH_TO_DATA
from hybrid_corres.utils import helper, visualization as vis
import scipy.io as sio
import matplotlib.pyplot as plt
from geop import linalg
import numpy as np

OBJ = '{}/faust/scans/{{0:03d}}_{{1:03d}}.obj'.format(PATH_TO_DATA)
MAT = '{}/faust/scans/{{0:03d}}_{{1:03d}}.mat'.format(PATH_TO_DATA)

for scan_id in range(100):
  for view_id in range(100):
    mat = sio.loadmat(MAT.format(scan_id, view_id))
    
    depth = mat['depth'].reshape(-1)
    points3d = mat['points3d'].reshape((-1, 3))
    height = int(mat['height'].reshape(-1))
    width = int(mat['width'].reshape(-1))
    valid_pixel_indices = mat['valid_pixel_indices'].reshape((-1, 2))
    intrinsic = mat['intrinsic']
    extrinsic = mat['extrinsic']

    """ Depth Images """
    print('Looking at Depth Images')
    stacked = helper.depth2image(depth, points3d, valid_pixel_indices, height, width)
    depth_image = stacked[:, :, 0]
    
    plt.imshow(depth_image)
    plt.show()

    """ Point Clouds """
    print('checking consitency between point cloud and depth images') 
    points = linalg.depth2pointcloud(depth_image, extrinsic, intrinsic)
    pcd = helper.read_obj(OBJ.format(scan_id, view_id))
    print('should see three point clouds fully overlap')
    vis.visualize_points([points, points3d, np.array(pcd.points)])
    pixels = linalg.pointcloud2pixel(points3d, extrinsic, intrinsic)
    pixels = pixels.astype(np.int64)
    depth_image_from_pcd = np.zeros((height, width))
    depth_image_from_pcd[(pixels[:, 0], pixels[:, 1])] = depth
    depth_image_copy = depth_image
    depth_image_copy[(pixels[:, 0], pixels[:, 1])] = depth
    concat = np.concatenate([depth_image_from_pcd, depth_image, depth_image_copy], axis=1)
    plt.imshow(concat)
    plt.show()
    #import ipdb; ipdb.set_trace()

    print('hey')
