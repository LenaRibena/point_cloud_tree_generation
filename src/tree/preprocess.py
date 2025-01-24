from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tqdm
from hydra.utils import to_absolute_path

from tree.utils.preprocess_utils import generate_cylinder_points


class DataPreprocessor:
    def preprocess(raw_data_path: Path, output_folder: Path, *, recursive_search: bool = False):
        raise NotImplementedError("Subclasses must implement this method")


class UrbanTreeDataPreprocessor(DataPreprocessor):
    def __init__(self, conf) -> None:
        self.conf = conf

    def sample_cylinders(self, csv_file: str | Path, total_points: int, nr_cylinders: int = 1024) -> np.ndarray:
        """
        Generate a point cloud from cylinders described in a CSV file.

        Parameters:
            csv_file (str): Path to the CSV file.
            total_points (int): Total number of points to sample.

        Returns:
            np.array: A Nx3 array of sampled points.
        """
        # Load the cylinder data
        df = pd.read_csv(csv_file)
        df = df.head(nr_cylinders)

        # Calculate total surface area for all cylinders
        df["surface_area"] = 2 * np.pi * df["radius"] * df["length"]

        # Determine the number of points per cylinder proportionally to its surface area
        df["scaled_area"] = np.sqrt(df["surface_area"])
        df["num_points"] = (df["scaled_area"] / df["scaled_area"].sum() * total_points).astype(int)

        # Calculate the rounding error
        difference = total_points - df["num_points"].sum()

        # Distribute the remaining points
        if difference != 0:
            # Sort cylinders by scaled_area fractional part (largest errors first)
            adjustment_indices = np.argsort(-(df["scaled_area"] % 1))

            # Apply the adjustment in one go
            adjustment_mask = adjustment_indices[: abs(difference)]
            df.loc[adjustment_mask, "num_points"] += np.sign(difference)

        point_cloud = []
        for _, row in df.iterrows():
            if row["num_points"] > 0:
                start = np.array([row["start_x"], row["start_y"], row["start_z"]])
                axis = np.array([row["axis_x"], row["axis_y"], row["axis_z"]])
                points = generate_cylinder_points(start, axis, row["length"], row["radius"], row["num_points"])
                point_cloud.append(points)

        # Combine all points into a single array
        point_cloud = np.vstack(point_cloud)

        # If the total number of points exceeds the target, randomly downsample
        if len(point_cloud) > total_points:
            indices = np.random.choice(len(point_cloud), total_points, replace=False)
            point_cloud = point_cloud[indices]

        return point_cloud

    def preprocess(self, raw_data_path: str | Path, output_folder: str | Path, *, recursive_search: bool = False):
        # Convert to Path objects
        raw_data_path = Path(raw_data_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        csv_files = raw_data_path.rglob("*.csv") if recursive_search else raw_data_path.glob("*.csv")
        if not csv_files:
            raise ValueError("No CSV files found in the specified path")
        elif recursive_search:
            csv_files = list(csv_files)
            if len(set([csv_file.stem for csv_file in csv_files])) != len(csv_files):
                raise ValueError("Duplicate file names found, please ensure all file names are unique!")

        for csv_file in tqdm.tqdm(csv_files, desc="Processing CSV files"):
            sampled_points = self.sample_cylinders(
                csv_file, total_points=self.conf.nr_points, nr_cylinders=self.conf.nr_cylinders
            )
            np.save(output_folder / f"{csv_file.stem}.npy", sampled_points)
        print("Preprocessing completed, saved to:", output_folder)


@hydra.main(version_base="1.2", config_path=to_absolute_path("configs"), config_name="preprocess")
def main(conf):
    match conf.dataset_name:
        case "urban_tree":
            preprocessor = UrbanTreeDataPreprocessor(conf.urban_tree_conf)
            preprocessor.preprocess(conf.raw_data_path, conf.output_folder, recursive_search=conf.recursive_search)
        case _:
            raise ValueError(f"Invalid dataset name: {conf.dataset_name}")


if __name__ == "__main__":
    main()
