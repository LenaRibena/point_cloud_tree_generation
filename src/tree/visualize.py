import open3d as o3d


def view_tree(points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([cloud])
