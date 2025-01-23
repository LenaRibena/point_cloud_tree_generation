import json
"""
Script to parse the start_script.sh file into the config.json file
"""
with open('start_script.sh', 'r') as file:
    start_script = file.read()

with open('config.json', 'r') as file:
    batch_config = json.load(file)

batch_config['taskGroups'][0]['taskSpec']['runnables'][0]['script']['text'] = start_script

with open('config.json', 'w') as file:
    json.dump(batch_config, file, indent=4)