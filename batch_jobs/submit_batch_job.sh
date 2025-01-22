#!/bin/bash
python parse_start_script_to_batch_config.py

# Change YOURJOBNAME to the name of the job you want to submit
gcloud batch jobs submit YOURJOBNAME --location europe-west1 --config config.json