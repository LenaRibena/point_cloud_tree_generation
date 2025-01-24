import numpy as np
import requests

from tree.visualize import view_tree

if __name__ == "__main__":
    # First run the following from terminal
    # conda activate ml-ops
    # cd C:\Users\Jason\OneDrive - Danmarks Tekniske Universitet\Masters\1_Semester\ML-Ops\point_cloud_tree_generation
    # uvicorn --reload --port 8000 src.tree.app:app

    url = "http://localhost:8000/"

    response = requests.get(url + "generate/flow")
    tree = np.array(response.json()["tree"])
    view_tree(tree)

    response = requests.get(url + "generate/gauss")
    tree = np.array(response.json()["tree"])
    view_tree(tree)
