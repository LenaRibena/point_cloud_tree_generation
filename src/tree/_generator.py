import hydra
import open3d as o3d
from hydra.utils import to_absolute_path

from tree.models.vae_flow import FlowVAE


def view_tree(points):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.visualization.draw_geometries([cloud])


@hydra.main(version_base="1.2", config_path=to_absolute_path("configs"), config_name="train")
def main(args):
    model = FlowVAE.load("models/flow_model.pth", args)
    view_tree(model.generate())


if __name__ == "__main__":
    main()
