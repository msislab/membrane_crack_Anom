import os, glob, tqdm
import cv2
import argparse
import numpy as np
from visualize import drawBox


refPt = []
cropping = False

def visGT(path=None, img=None, obb=False, boxType='xywh'):
    if img is not None:
         _img = img
    else:     
        _img = cv2.imread(path)
    labelPath = path.split('.jpg')[0] + '.txt'
    boxes = []
    if obb:
        with open(labelPath, 'r') as f:
            for line in f.readlines():
                boxStr = line.split()
                box = [int(boxStr[0]),float(boxStr[1])*_img.shape[1], float(boxStr[2])*_img.shape[0], 
                            float(boxStr[3])*_img.shape[1], float(boxStr[4])*_img.shape[0],
                            float(boxStr[5])*_img.shape[1], float(boxStr[6])*_img.shape[0],
                            float(boxStr[7])*_img.shape[1], float(boxStr[8])*_img.shape[0]]
                boxes.append(np.array(box))
        boxes = np.array(boxes)        
        _img  = drawBox(_img, boxes, obb=obb, thickness=1)

        cv2.imshow('',_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        with open(labelPath, 'r') as f:
            for line in f.readlines():
                boxStr = line.split()
                box = [int(boxStr[0]),float(boxStr[1])*_img.shape[1], float(boxStr[2])*_img.shape[0], 
                        float(boxStr[3])*_img.shape[1], float(boxStr[4])*_img.shape[0]]
                boxes.append(np.array(box))
        boxes = np.array(boxes)        
        _img  = drawBox(_img, boxes)

    # cv2.imshow('',_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return _img

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, _refPt
    
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		_refPt = (x, y)
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append(_refPt)
		cropping = False
		# # # draw a rectangle around the region of interest
		# cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
		# cv2.imshow("image", param)

def argParser():
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--dataPath', type=str,
                        default=None,
                        help='specify the original data dir path')
    parser.add_argument('--savePath', type=str,
                        default=None,
                        help='specify the save dir path to save modified data')
    
    args = parser.parse_args()
    return args

def resetLabels(_path, ROI, deltaPoints, savePath):
    #  cv2.imshow('',ROI)
    #  cv2.waitKey()
    #  cv2.destroyAllWindows()
    labelPath = _path.split('.')[0]+'.txt'
    name      = _path.split('/')[-1]
    with open(labelPath, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        label = line.split()
        _label = [int(label[0]), float(label[1]), float(label[2]),
                  float(label[3]), float(label[4]), float(label[5]),
                  float(label[6]), float(label[7]), float(label[8])]
        labels.append(_label)
    labels = np.array(labels)
    try:
        # scale back
        labels[:,1::2] *= 1920
        labels[:,2::2] *= 1280
        # find chnage due to crop    
        # delta_x = 1920 - ROI.shape[1]
        # delta_y = 1280 - ROI.shape[0]
        # adjust according to change
        labels[:,1::2] -= deltaPoints[0]
        labels[:,2::2] -= deltaPoints[1]
        # clip outer coordinates to fit inside the image
        labels[:,1::2] = np.clip(labels[:,1::2], 0, ROI.shape[1]-1)
        labels[:,2::2] = np.clip(labels[:,2::2], 0, ROI.shape[0]-1)
        # scal with new dims
        labels[:,1::2] /= ROI.shape[1]
        labels[:,2::2] /= ROI.shape[0]
    except:
        pass
    newLabels = []
    for label in labels:
        line = f'{int(label[0])} {label[1]} {label[2]} {label[3]} {label[4]} {label[5]} {label[6]} {label[7]} {label[8]} \n'
        newLabels.append(line)
    labelName = savePath + '/' + name.split('.')[0] + '.txt'
    imgName   = savePath + '/' + name
    ROI = cv2.resize(ROI, (1920, 1280))
    with open(labelName, 'w') as f:
        f.writelines(newLabels)
    cv2.imwrite(imgName, ROI)    
    # print()

    # visGT(path=imgName, obb=True)
    # print()

def main():
    args = argParser()
    filpaths = glob.glob(f'{args.dataPath}/*.jpg')
    ind = 0
    # num = 0
    for file in tqdm.tqdm(filpaths):
        global refPt
        img = cv2.imread(file)
        # img = cv2.resize(img, (1280, 1080), interpolation=cv2.INTER_AREA)
        clone = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)
        
        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img = clone.copy()
                # refPt = []
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        cv2.destroyAllWindows()
        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(refPt) >= 2:
            pts = [refPt[ind], refPt[ind+1]]
            ind+=2
            roi = clone[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
            resetLabels(file,roi, pts[0], args.savePath)
            # roi = cv2.resize(roi, (1920,1280), interpolation=cv2.INTER_LINEAR)
            # imgPath = args.savePath + '/' + f'{str(num).zfill(3)}.jpg'
            # num+=1
            # cv2.imwrite(imgPath, roi)
            # cv2.imshow("ROI", roi)
            # cv2.waitKey(0)
        # cv2.imshow('original data', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()    
    print()

if __name__=='__main__':
    main()