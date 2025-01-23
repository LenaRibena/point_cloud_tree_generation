# Install nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Authenticate and run container image with gpu
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin europe-west1-docker.pkg.dev
#todo: insert giving arguments for the experiment
docker run -e WANDB_PROJECT=$WANDB_PROJECT \
  -e WANDB_ENTITY=$WANDB_ENTITY \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --volume /mnt/disks/data-tree/processed-data:/trees/data/processed/urban_tree_dataset \
  --volume /mnt/disks/models:/trees/models \
  --gpus all \
  --entrypoint /bin/bash europe-west1-docker.pkg.dev/dtu-mlops-tree/tree/train:latest -c "wandb sweep configs/sweep.yaml"
