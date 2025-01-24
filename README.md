# tree

## Project description

Modelling objects is a tedious, yet essential, process. An example of its prevalence is in 3D video game background modeling which requires vast knowledge of object anatomy and variation across species. There is therefore an incentive to leverage this process; especially for ubiquitous objects such as different types of trees. This project therefore aims at generating point clouds representing trees.

[The synthetic tree point cloud dataset](https://springernature.figshare.com/collections/_/6788358) provides pointclouds from 40 scanning projects on the streets of  Munich. The dataset includes a total of 3755 leaf-off individual point clouds of trees and processed tree quantative models using the algorithm in [TreeQSM](https://github.com/InverseTampere/TreeQSM).

<img src="figures/point_cloud.png" alt="drawing" width="300"/>
<img src="figures/TreeQSM.png" alt="drawing" width="300"/>

*Image description: A data sample from the dataset. It contains the point cloud (left) along with the corresponding quantative structure model with the fitted cylinders. (right)*

The model chosen to tackle this particular task is the one presented in [Diffusion Probabilistic Models for 3D Point Cloud Generation](https://arxiv.org/abs/2103.01458) where a PointNet is trained to encode pointclouds into a sort of shape latent space distribution which the diffusion model conditions on and denoises back into a point cloud. When generating new trees, a gaussian $N(0,I)$ is diffused.

Since the tree pointcloud dataset had 100.000s of points, they were downsampled by using the corresponding TreeQSM model, filtering out cylinders with low radius and prioritising branch orders.
The downsampled tree from the above images would for example look like this:
<img src="figures/point_cloud_downsampled.png" alt="drawing" width="300"/>

## Project Canvas

*This is the project canvas for the first iteration after week 1, a lot of changed since then*
<img src="figures/canvas.png" alt="drawing" width="600"/>


## Project structure

The directory structure of the project looks like this:
```txt
├── .devcontainer/
├── .dvc/
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│   │   └── tests.yaml
├── batch_jobs/
├── configs/                  # Configuration files
│   ├── preprocess.yaml
│   ├── sweep.yaml
│   └── train.yaml
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│   │   └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   └── project_name/
│   │   ├── modules/          # Models modules
│   │   │   └── encoders/
│   │   │   │   ├── __init__.py
│   │   │   │   └── pointnet.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── preprocess_utils.py
│   │   │   └── train_utils.py
│   │   ├── app_client.py
│   │   ├── app.py
│   │   ├── data.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── cloudbuild.yaml
├── LICENSE
├── models.dvc
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements_dev.txt      # Development requirements
├── requirements.txt          # Project requirements
└── tasks.py                  # Project tasks
└── test.sh
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
