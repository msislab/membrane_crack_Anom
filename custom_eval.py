import os, glob
import numpy as np
from visualize import stats_obb, calculate_iou, find_overlaps

gtPath = "/home/zafar/old_pc/data_sets/robot-project-datasets/TML-Bur-detection_v1/val"
predPath = "/home/zafar/membrane_crack_Anom/runs/TML-bur/val6/labels"

anomaly = 0
for predFile in glob.glob(f'{predPath}/*.txt'):
    name = predFile.split('/')[-1]
    _gtPath = gtPath + f'/{name}' 
    with open(predFile, 'r') as f:
        predLines = f.readlines()
        predBoxes = []
    for line in predLines:
        boxStr = line.split()
        box = [int(boxStr[0]),float(boxStr[1])*1920, float(boxStr[2])*1280, 
                float(boxStr[3])*1920, float(boxStr[4])*1280,
                float(boxStr[5])*1920, float(boxStr[6])*1280,
                float(boxStr[7])*1920, float(boxStr[8])*1280]
        predBoxes.append(np.array(box))
    predBoxes  = np.array(predBoxes)
    with open(_gtPath, 'r') as f:
        gtLines = f.readlines()
        gtBoxes = []
    for line in gtLines:
        boxStr = line.split()
        box = [int(boxStr[0]),float(boxStr[1])*1920, float(boxStr[2])*1280, 
                float(boxStr[3])*1920, float(boxStr[4])*1280,
                float(boxStr[5])*1920, float(boxStr[6])*1280,
                float(boxStr[7])*1920, float(boxStr[8])*1280]
        gtBoxes.append(np.array(box))
    gtBoxes  = np.array(gtBoxes)
    tp, fp = find_overlaps(gtBoxes, predBoxes, 0.3, True)
    if (tp/len(predBoxes) > 0.8):
        anomaly += 1

acc=anomaly/47
print(acc)        



