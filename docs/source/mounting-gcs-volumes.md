# Guide to mount GCS Buckets with rclone

## Windows
1. Download [WinFSP](https://winfsp.dev/rel/) 
2. Download [rclone](https://rclone.org/downloads/) 
3. Navigate to the folder with rclone and run `rclone config`
4. Configure rclone to gcloud storage by following the steps shown [here](https://rclone.org/googlecloudstorage/)
    - When writing the project number, you can check the project numbers by running `gcloud projects list`
    - For practical purposes, please use the remote name `dtu-mlops-tre`
    - Make sure to use the multiregional eu for region
5. Mount the whatever you need, for example:
`rclone mount dtu-mlops-tree:data-tree/urban_tree_dataset C:\University\ML_ops\point_cloud_tree_generation\data\raw\urban_tree_dataset`

## Mac
1. Download MacFUSE `brew install --cask macfuse`
2. Enable [support for third party kernel extensions](https://github.com/macfuse/macfuse/wiki/Getting-Started)
3. Download [rclone](https://rclone.org/downloads/)
4. Navigate to the downloaded rclone folder and start the config by running `./rclone config`
5. Configure rclone to gcloud storage by following the steps shown [here](https://rclone.org/googlecloudstorage/)
    - When writing the project number, you can check the project numbers by running `gcloud projects list`
    - For practical purposes, please use the remote name `dtu-mlops-tre`
    - Make sure to use the multiregional eu for region
6. Mount the whatever you need, for example:
`rclone mount dtu-mlops-tree:data-tree/urban_tree_dataset C:\University\ML_ops\point_cloud_tree_generation\data\raw\urban_tree_dataset`


## Linux (can be done directly with gcsfuse)
1. Follow https://cloud.google.com/storage/docs/cloud-storage-fuse/mount-bucket#mount-bucket