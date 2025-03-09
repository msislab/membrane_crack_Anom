import os
import tqdm
import json

json_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".json")]

config ={}
for file in tqdm.tqdm(json_files):
    key = file.split('.json')[0]
    filePath = os.path.join(os.getcwd(), file)

    print(f"Loading JSON file: {filePath}")
    
    try:
        with open(filePath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error parsing {filePath}: {e}")
    config[f"{key}"] = data

with open("brightness_config.json", "w", encoding="utf-8") as json_file:
    json.dump(config, json_file, indent=4)

print()