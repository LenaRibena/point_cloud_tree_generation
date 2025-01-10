# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import open3d as o3d # pip install open3d

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from IPython.display import display


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def o3d_cloud(points, colour=None, colours=None, normals=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)
    if colour is not None:
        return cloud.paint_uniform_color(colour)
    elif colours is not None:
        cloud.colors = o3d.utility.Vector3dVector(colours)

    return cloud


def o3d_line_set(vertices, edges, colour=None):
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(edges)
    )
    if colour is not None:
        return ls.paint_uniform_color(colour)
    return ls


def o3d_path(vertices, colour=None):
    idx = np.arange(vertices.shape[0] - 1)
    edge_idx = np.column_stack((idx, idx + 1))
    if colour is not None:
        return o3d_line_set(vertices, edge_idx, colour)
    return o3d_line_set(vertices, edge_idx)


def o3d_merge_linesets(line_sets, colour=(0, 0, 0)):
    sizes = [np.asarray(ls.points).shape[0] for ls in line_sets]
    offsets = np.cumsum([0] + sizes)

    points = np.concatenate([ls.points for ls in line_sets])
    idxs = np.concatenate([ls.lines + offset for ls, offset in zip(line_sets, offsets)])

    return o3d_line_set(points, idxs).paint_uniform_color(colour)

@dataclass
class BranchSkeleton:
    _id: int
    parent_id: int
    xyz: np.array
    radii: np.array
    child_id: int = -1
        
    def to_o3d_lineset(self, colour=(0, 0, 0)):
        return o3d_path(self.xyz, colour)    
        
@dataclass
class Cloud:
    xyz: np.array
    rgb: np.array
    class_l: np.array = None
    medial_vector: np.array = None
        
    def as_open3d(self):
        return o3d_cloud(self.xyz, colours=self.rgb)

@dataclass
class TreeSkeleton:
    _id: int
    branches: Dict[int, BranchSkeleton]
         
    def as_o3d_lineset(self):
        return o3d_merge_linesets([branch.to_o3d_lineset() for branch in self.branches.values()])






def unpackage_data(data: dict) -> Tuple[Cloud, TreeSkeleton]:
    tree_id = data["tree_id"]
    branch_id = data["branch_id"]
    branch_parent_id = data["branch_parent_id"]
    skeleton_xyz = data["skeleton_xyz"]
    skeleton_radii = data["skeleton_radii"]
    sizes = data["branch_num_elements"]

    medial_vector = data.get("medial_vector", data.get("vector", None))

    cld = Cloud(
        xyz=data["xyz"],
        rgb=data["rgb"],
        class_l=data["class_l"],
        medial_vector=medial_vector,
    )

    offsets = np.cumsum(np.append([0], sizes))

    branch_idx = [np.arange(size) + offset for size, offset in zip(sizes, offsets)]
    branches = {}

    for idx, _id, parent_id in zip(branch_idx, branch_id, branch_parent_id):
        branches[_id] = BranchSkeleton(
            _id, parent_id, skeleton_xyz[idx], skeleton_radii[idx]
        )

    return cld, TreeSkeleton(tree_id, branches)


def load_data_npz(path: Path) -> Tuple[Cloud, TreeSkeleton]:
    return unpackage_data(np.load(str(path)))


def main():
    cloud, skeleton = load_data_npz("data/raw/london_14.npz")
    o3d_pcd = cloud.as_open3d()
    o3d_skeleton = skeleton.as_o3d_lineset()


    o3d.visualization.draw_geometries([o3d_pcd, o3d_skeleton])

if __name__ == "__main__":
    main()