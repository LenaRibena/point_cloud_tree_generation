# Authenticate and run container image with gpu
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin europe-west1-docker.pkg.dev
#todo: insert giving arguments for the experiment
docker run -e WANDB_PROJECT=$WANDB_PROJECT \
  -e WANDB_ENTITY=$WANDB_ENTITY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --volume /mnt/disks/data-tree/processed-data:/trees/data/processed/urban_tree_dataset \
  --volume /mnt/disks/models:/trees/models \
  --entrypoint /bin/bash europe-west1-docker.pkg.dev/dtu-mlops-tree/tree/train:latest -c "python -m tree.train epochs=100 model=gaussian experiment_dir=/trees/models"
