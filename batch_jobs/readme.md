# How to set up batch jobbing trainings

This is a small guide to do batch job training using a docker image. If you want to do the training outside the docker image, you can either submit batch jobs with custom images/bootDisks/mounted volumes (so you can have the files you want) or change start_script.sh to set up the VM for training from the bottom. 

Since Google Batch is a little weird with the Container Optimized OS option where you 
cannot run the docker images with custom commands, then it's impossible to run any dockers
with GPU or any custom command.

## Files
`config.json`: Batch job config file, here you can specify the resources you need etc. See https://cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs#resource:-job for more info
`start_script.sh`: Start up script for the batch VM to run outside the Docker container. nvidia-container-toolkit and docker gcloud authentication is done inside here.
`parse_start_script_to_batch_config.py`: parses the content of `start_script.sh` to config.json in the script: {text: ""} field. 
`submit_batch_job.sh`: Calls parse_start_script_to_batch_config.py and submits the batch job. Specify a batch job name inside here (or just change it so you can run the bash script with arguments...)
`examples/`: Folder to show example batch job.

## Steps to submit a batch job for training
1. Make sure your docker image has all the needed commands and files for training
2. Submit it to GCP Artifact Registry. [Guide Here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#pushing)
3. Change [start_script.sh](start_script.sh) if you need other commands before running the docker container or if you want to change the docker run command.
4. Change the batch job configs to what you need inside config.json (ignore the `script` part, [start_script.sh](start_script.sh) will automatically be parsed into there)
6. Change the batch job name inside [submit_batch_job.sh](submit_batch_job.sh)
5. Run ```bash submit_batch_job.sh```
6. You can follow the job's status and progress inside Google Cloud Batch. Outputs and prints will be visible inside the logs.

## Detailed example
1. Create python script to test if python can access gpu inside the docker container and a corresponding requirements.txt file.
See [test.py](example/test.py)
```python
import torch
print('Testing if pytorch can see GPU...')
print(torch.cuda.is_available())
```

2. Running multiple commands and displaying python prints can be done by creating a bash script for the container to run as its entrypoint:
See [docker_run.py](example/docker_run.py)
```bash
#!/bin/bash
nvidia-smi
python test.py
```

3. Create the Dockerfile and copy relevant files into it:
```dockerfile
FROM python:3.10

COPY requirements.txt requirements.txt
COPY test.py test.py
COPY docker_run.sh docker_run.sh
RUN pip install -r requirements.txt

RUN chmod +x /docker_run.sh

ENTRYPOINT ["/docker_run.sh"]
```

4. Build the docker inside the [example](example) folder
```shell
cd example
docker build -t batch_test .
```

5. Retag the image
```shell
docker tag batch_test europe-west1-docker.pkg.dev/bluemar-code/biodiversity-test/simple_image:latest
```

6. Go into GCP Artifact Registry and create a repository [Arifact Registry link for bluemar-code project here](https://console.cloud.google.com/artifacts?referrer=search&project=bluemar-code). You can just keep the default configs and change `Region` to `europe-west1` for example.
![create_repo_img_with_way_too_many_arrows](images/create_repository.png)

7. Push the retagged image to the newly created Artifact Registry repository that was named `biodiversity-test`here with the region `europe-west1` where the image name is changed to `simple_image` for the push.
```shell
docker push batch_test europe-west1-docker.pkg.dev/bluemar-code/biodiversity-test/simple_image
```

8. Create the [start_script.sh](start_script.sh) to include the minimum of downloading nvidia-container-toolkit (needed to run docker containers with gpu) and authenticating docker to pull from GCP Artifact Registry.
See [start_script.sh](start_script.sh)
```bash
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
docker run --gpus all europe-west1-docker.pkg.dev/bluemar-code/biodiversity-test/simple_image:latest

```

9. No need for a lot of resources for the example so the batch job config file [config.json](config.json) is set to:
```yaml
{
    "taskGroups": [
        {
            "taskSpec": {
                "runnables": [
                    {
                        "script": {
                            "text": ""
                        }
                    }
                ],
                "computeResource": {
                    "cpuMilli": 2000,
                    "memoryMib": 16
                },
                "maxRetryCount": 0,
                "maxRunDuration": "360s"
            },
            "taskCount": 1,
            "parallelism": 1
        }
    ],
    "allocationPolicy": {
        "instances": [
            {
                "installGpuDrivers": true,
                "policy": {
                    "machineType": "n1-standard-2",
                    "accelerators": [
                        {
                            "type": "nvidia-tesla-t4",
                            "count": 1
                        }
                    ]
                }
            }
        ]
    },
    "logsPolicy": {
        "destination": "CLOUD_LOGGING"
    }
}
```

10. Now lastly, submit the batch job by running:
```shell
bash submit_batch_job.sh
```

11. The progress of the batch job can be followed inside GCP Batch at https://console.cloud.google.com/batch where you can view the logs of the job. 

12. Check that the batch job sucessfully ran a docker container with gpus visible for the python script inside the container through the logs: 
![batch_success](images/batch_complete.png)
Remember that [docker_run.sh](example/docker_run.sh)
```bash
#!/bin/bash
nvidia-smi
python test.py
```
was run inside the container and the logs show that both commands worked!

13. Done!

