import os, glob, tqdm
import cv2
import argparse


refPt = []
cropping = False

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

def main():
    args = argParser()
    filpaths = glob.glob(f'{args.dataPath}/*.jpg')
    ind = 0
    num = 83
    for file in tqdm.tqdm(filpaths):
        global refPt
        img = cv2.imread(file)
        img = cv2.resize(img, (1280, 1080), interpolation=cv2.INTER_AREA)
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
        # if there are two reference points, then crop the region of interest
        # from the image and display it
        if len(refPt) >= 2:
            pts = [refPt[ind], refPt[ind+1]]
            ind+=2
            roi = clone[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]
            roi = cv2.resize(roi, (640,640), interpolation=cv2.INTER_LINEAR)
            imgPath = args.savePath + '/' + f'{str(num).zfill(3)}.jpg'
            num+=1
            cv2.imwrite(imgPath, roi)
            # cv2.imshow("ROI", roi)
            # cv2.waitKey(0)
        # cv2.imshow('original data', img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()    
    print()

if __name__=='__main__':
    main()