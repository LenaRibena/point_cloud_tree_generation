import numpy as np
import open3d as o3d


def view_tree(points: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.paint_uniform_color([0.1, 0.8, 0.2])
    o3d.visualization.draw_geometries([cloud])
