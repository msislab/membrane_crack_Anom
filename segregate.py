import os 
import glob 
import shutil
import cv2
import tqdm
files = glob.glob("/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/20250508-A/1/*/*/Inputs/*.webp", recursive=True)
# files = glob.glob("/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/4-6_lineA/*/*/Inputs/*.png", recursive=True)
surfaces = ["Front-pin_auto_0", "Front-pin_auto_1", "Front-pin-2nd_auto_0", "Front-pin-2nd_auto_1", "Top-pin_auto_0", "Top-pin_auto_1", "Top-pin-2nd_auto_1", "Top-pin-2nd_auto_0"]



dest_path = "/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/20250508-A/1"
for i, surface in enumerate(surfaces):
    des_folder = os.path.join(dest_path, f"{surface}")
    os.makedirs(des_folder, exist_ok=True)

for i, file in enumerate(tqdm.tqdm(files)):
    connector_id = file.split("/")[-3]
    name = file.split("/")[-1]
    if "Front-pin_auto_0" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')  # Remove leading zeros
        name = os.path.join(dest_path, "Front-pin_auto_0", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Front-pin_auto_0", f"{connector_id}_{os.path.basename(file)}"))
    elif "Front-pin_auto_1" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Front-pin_auto_1", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Front-pin_auto_1", f"{connector_id}_{os.path.basename(file)}"))
    elif "Front-pin-2nd_auto_0" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Front-pin-2nd_auto_0", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Front-pin-2nd_auto_0", f"{connector_id}_{os.path.basename(file)}"))
    elif "Front-pin-2nd_auto_1" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Front-pin-2nd_auto_1", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Front-pin-2nd_auto_1", f"{connector_id}_{os.path.basename(file)}"))
    elif "Top-pin-2nd_auto_0" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Top-pin-2nd_auto_0", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Top-pin-2nd_auto_0", f"{connector_id}_{os.path.basename(file)}"))
    elif "Top-pin-2nd_auto_1" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Top-pin-2nd_auto_1", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Top-pin-2nd_auto_1", f"{connector_id}_{os.path.basename(file)}"))
    elif "Top-pin_auto_0" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Top-pin_auto_0", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Top-pin_auto_0", f"{connector_id}_{os.path.basename(file)}"))
    elif "Top-pin_auto_1" in file:
        img = cv2.imread(file)
        connector_id = connector_id.lstrip('0')
        name = os.path.join(dest_path, "Top-pin_auto_1", f"{name}-{connector_id}.png")
        cv2.imwrite(name, img)
        # shutil.copy(file, os.path.join(dest_path, "Top-pin_auto_1", f"{connector_id}_{os.path.basename(file)}"))