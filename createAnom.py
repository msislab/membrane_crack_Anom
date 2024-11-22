import cv2, os, glob, tqdm, argparse, math
from natsort import natsorted
import random
import numpy as np
from getROI import GetROI


def generateScar(mask, scar_height, scar_width, y_start, x_start, irregularity=0.2, scartype = None):
    
    if scartype is None:
        # scartype = random.randint(0,3)
        scartype = 0
    # scartype=0
    if scartype==0:
        idx = 0
        for y in range(y_start, scar_height):
            # if idx==0 or idx%3==0:
            x_offset = int((random.random()-0.4) * irregularity * scar_width)
            start_x  = max(0, x_start + x_offset)
            end_x    = start_x + scar_width
            mask[y, start_x:end_x] = 255
    
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

    # cv2.imshow('', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    kernel  = np.ones((3, 3), np.uint8)
    # _kernel = np.ones((2,2), np.uint8)
    # if scar_width<4:
    #     # mask = cv2.GaussianBlur(mask, (5,5),0)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel)
    #     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if scar_width>6:
        mask = cv2.GaussianBlur(mask, (3,3),0)
        mask = cv2.erode(mask, kernel)

    return mask             # cv2.GaussianBlur(mask, (3,3),0)    
    
def process(img, anomRoi, scarRoi, imgRoi):
    cv2.imshow('', np.hstack((anomRoi, scarRoi)))
    cv2.waitKey()
    cv2.destroyAllWindows()
    height, width, _ = anomRoi.shape
    
    if width>height:
        scar_width = random.randint(2, 8)        # width//12
        if scar_width > 3:
            scar_height = random.randint(15, height-5)
        else:
            scar_height = random.randint(8, height//2)

        x_start   = random.randint((width//4), (width//2)+(width//4))
        y_start   = 1
    elif height>width: 
        scar_width  = random.randint(8, width//1.5)
        scar_height = random.randint(2, 15)
        x_start     = 1
        y_start     = random.randint((height//4), (height//2)+(height//4))

    scar_mask = np.zeros_like(anomRoi, dtype=np.uint8)
    scar_mask = generateScar(scar_mask, scar_height, scar_width, y_start, x_start, irregularity=0.7)
    # cv2.imshow('', scar_mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    scar_mask_boolean= scar_mask>0
    scar_mask        = (scarRoi*scar_mask_boolean)
    img[imgRoi[0][1]:imgRoi[1][1], imgRoi[0][0]:imgRoi[1][0]] = \
        img[imgRoi[0][1]:imgRoi[1][1], imgRoi[0][0]:imgRoi[1][0]] \
        *~scar_mask_boolean + (scar_mask*scar_mask_boolean)
    
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img

def pasteAnomaly(img, region1, region2, region3):

    idx = random.randint(1,3)
    if idx==1:
        # adding anomaly in region1 only (below region)
        anom1_roi   = img[region1[0][1]:region1[1][1], region1[0][0]:region1[1][0]]     # for white scar anomaly
        whiteRegion = img[region1[0][1]-150:region1[1][1]-150, region1[0][0]:region1[1][0]] # white anomaly extraction region
        img         = process(img, anom1_roi, whiteRegion, region1)
    elif idx==2:
        # adding anomaly in region1
        anom1_roi    = img[region1[0][1]:region1[1][1], region1[0][0]:region1[1][0]]     # for white scar anomaly
        whiteRegion  = img[region1[0][1]-150:region1[1][1]-150, region1[0][0]:region1[1][0]] # white anomaly extraction region
        img          = process(img, anom1_roi, whiteRegion, region1)
        # adding anomaly in region2
        anom2_roi    = img[region2[0][1]:region2[1][1], region2[0][0]:region2[1][0]]     # for black scar anomaly 1
        blackregion1 = img[region2[0][1]-region2[0][1]:region2[1][1]-region2[0][1], region2[0][0]:region2[1][0]] # black anomaly extraction region1
        img          = process(img, anom2_roi, blackregion1, region2)
    elif idx==3:
        # adding anomaly in region1
        anom1_roi    = img[region1[0][1]:region1[1][1], region1[0][0]:region1[1][0]]     # for white scar anomaly
        whiteRegion  = img[region1[0][1]-150:region1[1][1]-150, region1[0][0]:region1[1][0]] # white anomaly extraction region
        img          = process(img, anom1_roi, whiteRegion, region1)
        # adding anomaly in region2
        anom2_roi    = img[region2[0][1]:region2[1][1], region2[0][0]:region2[1][0]]     # for black scar anomaly 1
        blackregion1 = img[region2[0][1]-region2[0][1]:region2[1][1]-region2[0][1], region2[0][0]:region2[1][0]] # black anomaly extraction region1
        img          = process(img, anom2_roi, blackregion1, region2)
        # adding anomaly in region3
        anom3_roi    = img[region3[0][1]:region3[1][1], region3[0][0]:region3[1][0]]     # for black scar anomaly 2
        blackregion2 = img[region3[0][1]:region3[1][1], region3[0][0]-80:region3[1][0]-80] # black anomaly extraction region2
        img          = process(img, anom3_roi, blackregion2, region3)

    # cv2.imshow('', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
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
    getroi    = GetROI()
    args      = parseArgs()
    filePaths = natsorted(glob.glob(f'{args.dataPath}/*.jpg'))
    # refImg = cv2.imread(filePaths[0])
    idx   = 0
    for file in tqdm.tqdm(filePaths):
        img = cv2.imread(file)
        roi_1 = getroi._getROI(img=img, windowName='Select ROI_1')
        roi_2 = getroi._getROI(img=img, windowName='Select ROI_2')
        roi_3 = getroi._getROI(img=img, windowName='Select ROI_3')
        _img = cv2.rectangle(img, (roi_1[0][0],roi_1[0][1]), (roi_1[1][0],roi_1[1][1]), (0,0,255), 2)
        _img = cv2.rectangle(img, (roi_2[0][0],roi_2[0][1]), (roi_2[1][0],roi_2[1][1]), (0,0,255), 2)
        _img = cv2.rectangle(img, (roi_3[0][0],roi_3[0][1]), (roi_3[1][0],roi_3[1][1]), (0,0,255), 2)
        cv2.imshow('', _img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        img = pasteAnomaly(img,roi_1,roi_2, roi_3)
        # cv2.imshow('',img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        imgPath = args.savePath + '/' + f'{str(idx).zfill(3)}.jpg'
        idx+=1
        # cv2.imwrite(imgPath, img)
    print()

if __name__=='__main__':
    main()