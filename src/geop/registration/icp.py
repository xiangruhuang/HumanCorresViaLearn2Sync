import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors as NN
from geometry import util as gutil

""" Iterative Closest Points (ICP) Method according to point-to-plane metric.
    Inputs:
        source: o3d.geometry.PointCloud
        target: o3d.geometry.PointCloud
        sigma: soft-thresholding [default 0.01]
        max_iter: maximum number of iterations [default 100]
        stopping_threshold: stopping threshold for ICP algorithm [default 1e-4]
    Outputs:
        transform: np.ndarray of shape [4, 4].
                   Transformation from source to target.
"""
def icp_reweighted(source, target, sigma=0.01, max_iter = 100,
                   stopping_threshold=1e-4):
    """ If target has no normals, estimate """
    if np.array(target.normals).shape[0] == 0:
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
                                        radius=0.2, max_nn=30)
        o3d.estimate_normals(target, search_param=search_param)

    tree = NN(n_neighbors=1, algorithm='kd_tree', n_jobs=10)
    tree = tree.fit(np.array(target.points))
    n = np.array(source.points).shape[0]
    normals = np.array(target.normals)
    points = np.array(target.points)
    weights = np.zeros(n)
    errors = []
    transform = np.eye(4)

    for itr in range(max_iter):
        p = np.array(source.points)
        R, trans = gutil.unpack(transform)
        p = (R.dot(p.T) + trans.reshape((3, 1))).T
        _, indices = tree.kneighbors(p)

        """ (r X pi + pi + t - qi)^T ni """
        """( <r, (pi X ni)> + <t, ni> + <pi-qi, ni> )^2"""
        """ (<(r; t), hi> + di)^2 """
        nor = normals[indices[:, 0], :]
        q = points[indices[:, 0], :]
        d = np.sum(np.multiply(p-q, nor), axis=1) #[n]
        h = np.zeros((n, 6))
        h[:, :3] = np.cross(p, nor)
        h[:, 3:] = nor
        weight = (sigma**2)/(np.square(d)+sigma**2)
        H = np.multiply(h.T, weight).dot(h)
        g = -h.T.dot(np.multiply(d, weight))
        delta = np.linalg.solve(H, g)
        errors = np.abs(d)
        print('iter=%d, delta=%f, mean error=%f, median error=%f' % (
                itr, np.linalg.norm(delta, 2),
                np.mean(errors), np.median(errors)))
        if np.linalg.norm(delta, 2) < stopping_threshold:
            break
        trans = delta[3:]
        R = gutil.rodrigues(delta[:3])
        T = gutil.pack(R, trans)
        transform = T.dot(transform)

    return transform

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='reweighted ICP algorithm')
    parser.add_argument('--source', type=str,
                        help='source point cloud or mesh in .ply format')
    parser.add_argument('--target', type=str,
                        help='target point cloud or mesh in .ply format')
    args = parser.parse_args()

    source = o3d.read_point_cloud(args.source)
    try:
        mesh = o3d.read_triangle_mesh(args.target)
        if np.array(mesh.triangles).shape[0] == 0:
            assert False
        v = np.array(mesh.vertices)
        tri = np.array(mesh.triangles)
        v1 = v[tri[:, 0], :]
        v2 = v[tri[:, 1], :]
        v3 = v[tri[:, 2], :]
        normals = np.cross(v1-v3, v2-v3)
        normals = (normals.T / np.linalg.norm(normals, 2, axis=1)).T
        centers = (v1+v2+v3)/3.0

        target = o3d.PointCloud()
        target.points = o3d.Vector3dVector(centers)
        target.normals = o3d.Vector3dVector(normals)
    except:
        target = o3d.read_point_cloud(args.target)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
                                        radius=0.2, max_nn=30)
        o3d.estimate_normals(target, search_param=search_param)

    transformation = icp_reweighted(source, target)
    source.transform(transformation)
    o3d.draw_geometries([source, target])
