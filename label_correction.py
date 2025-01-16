import cv2, os, argparse, tqdm, glob
import numpy as np
from natsort import natsorted
import copy

path = '/home/zafar/old_pc/data_sets/robot-project-datasets/TML-Bur-detection_v1/val'

files = glob.glob(f'{path}/*.jpg')

for file in files:
    img = cv2.imread(file.strip())
    h, w, _ = img.shape
    labelFile = file.split('.')[0] + '.txt'
    updatedLabels = []
    with open(labelFile, 'r') as f:
        labels = f.readlines()
    print('Filename: ', file)
    print('Number of GT labels: ', len(labels))
    for label in labels:
        _img = copy.deepcopy(img)
        label = label.split()[1:]

        x1 = int(float(label[0]) * w)
        x2 = int(float(label[2]) * w)
        x3 = int(float(label[4]) * w)
        x4 = int(float(label[6]) * w)

        y1 = int(float(label[1]) * h)
        y2 = int(float(label[3]) * h)
        y3 = int(float(label[5]) * h)
        y4 = int(float(label[7]) * h)

        cv2.circle(_img, (x1,y1), 2, (0,0,255), -1)
        cv2.circle(_img, (x2,y2), 2, (0,255,0), -1)
        cv2.circle(_img, (x3,y3), 2, (255,0,0), -1)
        cv2.circle(_img, (x4,y4), 2, (0,255,255), -1)

        # abs(x3-x1), abs(x4-x2), abs(y3-y1), abs(y4-y2)
        w1 = abs(x3-x1)
        w2 = abs(x4-x2)
        h1 = abs(y3-y1)
        h2 = abs(y4-y2)

        print(f'{w1 }', f'{w2 }', '\n', f'{h1 }', f'{h2 }')

        cv2.imshow('',_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


        print()
