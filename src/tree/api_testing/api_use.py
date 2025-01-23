import os

import numpy as np
import requests

from tree.visualize_tree import TreeViewer


def main():
    url = "http://localhost:8000/generate_tree/"
    response = requests.get(url)

    if response.status_code == 200:
        with open("tree.npy", "wb") as f:
            f.write(response.content)
        tree = np.load("tree.npy")
        tv = TreeViewer()
        tv.view(tree)

    else:
        print(f"Failed to get tree: {response.status_code}")


if __name__ == "__main__":
    os.chdir("src/tree/api_testing")  # Change to save own tree.npy
    main()
