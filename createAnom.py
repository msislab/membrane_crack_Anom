import cv2, os, glob, tqdm, argparse, math
from natsort import natsorted
import random
import numpy as np


def generateScar(mask, scar_height, scar_width, y_start, x_start, irregularity=0.2, scartype = None):
    
    if scartype is None:
        scartype = random.randint(0,3)
    # scartype=0
    if scartype==0:
        idx = 0
        for y in range(y_start, scar_height):
            # if idx==0 or idx%3==0:
            x_offset = int((random.random()-0.5) * irregularity * scar_width)
            start_x  = max(0, x_start + x_offset)
            end_x    = start_x + scar_width
            mask[y, start_x:end_x] = 255
            # idx+=1
        # if scar_width>4:
        #     kernel       = np.ones((3, 3), np.uint8)
        #     _kernel      = np.zeros((3,3), np.uint8)
        #     _kernel[:,1] = 1
        #     # kernel[:,1] = 1
        #     mask = cv2.erode(cv2.dilate(mask, _kernel), kernel)
            # mask = cv2.dilate(mask, kernel)
    
    elif scartype==1:
        x_start = x_start-10
        start_x = x_start
        idx = 0
        for y in range(y_start, scar_height):
            if idx==0 or idx%5==0:
                start_x  = int(start_x + irregularity)
                x_offset = int((random.random()) * irregularity * scar_width)
                start_x  = max(0, start_x + x_offset)
            end_x    = start_x + scar_width
            mask[y, start_x:end_x] = 255
            idx+=1
    
    elif scartype==2:
        x_start = x_start+10
        start_x = x_start
        idx=0
        for y in range(y_start, scar_height):
            start_x  = int(start_x - irregularity)
            # if idx==0 or idx%2==0:
            x_offset = int((random.random()) * irregularity * scar_width)
            start_x  = max(0, start_x + x_offset)
            end_x    = start_x + scar_width
            mask[y, start_x:end_x] = 255

    kernel  = np.ones((3, 3), np.uint8)
    _kernel = np.ones((2,2), np.uint8)
    if scar_width<4:
        # mask = cv2.GaussianBlur(mask, (5,5),0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif scar_width>5:
        mask = cv2.GaussianBlur(mask, (3,3),0)
        mask = cv2.erode(mask, kernel)

    return mask             # cv2.GaussianBlur(mask, (3,3),0)    
    

def pasteAnomaly(img, region1, region2):

    anom1_roi   = img[region1[0][1]:region1[1][1], region1[0][0]:region1[1][0]]     # for white scar anomaly
    anom2_roi   = img[region2[0][1]:region2[1][1], region2[0][0]:region2[1][0]]     # for black scar anomaly

    whiteRegion = img[region1[0][1]-150:region1[1][1]-150, region1[0][0]:region1[1][0]] # white anomaly extraction region
    # whiteRegion = whiteRegion*0.99
    darkRegion  = img[region2[0][1]-50:60, region2[0][0]:region2[1][0]]         # black anomaly extraction region
    
    height, width, _ = anom1_roi.shape
    scar_width       = random.randint(2, 7)        # width//12
    
    if scar_width > 3:
        scar_height = random.randint(10, height-5)
    else:
        scar_height = random.randint(4, height//2)

    x_start   = random.randint((width//3), (width//2)+15)
    y_start   = 5
    scar_mask = np.zeros_like(anom1_roi, dtype=np.uint8)
    scar_mask = generateScar(scar_mask, scar_height, scar_width, y_start, x_start, irregularity=0.9)

    scar_mask_boolean= scar_mask>0
    scar_mask        = (whiteRegion*scar_mask_boolean)
    img[region1[0][1]:region1[1][1], region1[0][0]:region1[1][0]] = \
        img[region1[0][1]:region1[1][1], region1[0][0]:region1[1][0]] \
        *~scar_mask_boolean + (scar_mask*scar_mask_boolean)    
    
    # cv2.imshow('', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print()
    return img

def parseArgs():
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--dataPath', type=str,
                        default=None,
                        help='specify the original data dir path')
    parser.add_argument('--savePath', type=str,
                        default=None,
                        help='specify the save dir path to save modified data')
    
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    filePaths = natsorted(glob.glob(f'{args.dataPath}/*.jpg'))
    # refImg = cv2.imread(filePaths[0])
    roi_1 = [(275,450), (435,550)]
    roi_2 = [(260,50), (410,150)]
    idx = 0
    for file in tqdm.tqdm(filePaths):
        img = cv2.imread(file)
        # img = cv2.rectangle(img, (275,450), (435,550), (0,0,255), 2)
        # img = cv2.rectangle(img, (260,50), (410,150), (0,255,0), 2)
        img = pasteAnomaly(img,roi_1,roi_2)
        imgPath = args.savePath + '/' + f'{str(idx).zfill(3)}.jpg'
        idx+=1
        cv2.imwrite(imgPath, img)
        
        # _img = cv2.rectangle(img, (115,260), (135,320), (255,0,0), 2)
        # _img = cv2.rectangle(img, (265,65), (405,150), (255,255,0), 2)
        # cv2.imshow('',_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    print()

if __name__=='__main__':
    main()