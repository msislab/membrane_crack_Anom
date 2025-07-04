import torch
import cv2
# import cupy as cp
import numpy as np
# from concurrent.futures import ThreadPoolExecutor
import time
# import copy
from loguru import logger
# import datetime
# from src.utils.colors import *
# import glob
import emoji
from colorama import Fore, Style 
from ultralytics import YOLO
import threading

# Configure logging
# logger.add(f"src/ai_vision/logs/m07_PinROI{datetime.datetime.now()}.log", rotation="10 MB", level="INFO")

class MODEL:
    """
    m07_PinROI model class. This class is implimented to get the desired foreground ROIs (Pin-ROI, Bur-ROI) of Pin surfaces.
    """
    def __init__(self, model_path='', confidence_threshold=0.25, 
                 height=1280, width=1920, warmup_runs=3, debug=False, 
                 device_id=0):  #TODO 3: Update the model_path, Add the parameters you need which can be modified from GUI
        """
        Initialize the model.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Confidence threshold for detections.
            warmup_runs (int): Number of dummy inference runs to warm up the model.
        """
        try:
            num_gpus        = torch.cuda.device_count()
            self.conf_thres = confidence_threshold
            self.device     = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and num_gpus > device_id else "cuda:0")
            self.h          = height
            self.w          = width
            self.logger     = logger
            self.kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self._kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.kernel1    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))

            self.model_lock1 = threading.Lock()
            self.model_lock2 = threading.Lock()
            self.model_lock3 = threading.Lock()
            self.model_lock4 = threading.Lock()

            self.model_1 = YOLO(model_path, task='obb').to(self.device)
            self.model_2 = YOLO(model_path, task='obb').to(self.device)
            self.model_3 = YOLO(model_path, task='obb').to(self.device)
            self.model_4 = YOLO(model_path, task='obb').to(self.device)
    
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(warmup_runs):
                _ = self.model_1.predict(source=dummy_image, conf=self.conf_thres, device=self.device, verbose=False)
                _ = self.model_2.predict(source=dummy_image, conf=self.conf_thres, device=self.device, verbose=False)
                _ = self.model_3.predict(source=dummy_image, conf=self.conf_thres, device=self.device, verbose=False)
                _ = self.model_4.predict(source=dummy_image, conf=self.conf_thres, device=self.device, verbose=False)
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.logger.exception(f"Error initializing model: {e}")
            raise e

    def processOBB(self, yoloOBB_rslt, conf_thres):
        # s = time.time()
        preds_list = []
        for result in yoloOBB_rslt:
            if result.obb is not None:
                boxes  = result.obb.xyxyxyxy.cpu().numpy().reshape(-1,8)
                clsz   = result.obb.cls.cpu().numpy()
                conf   = result.obb.conf.cpu().numpy()
                _preds = np.hstack([clsz[:,None], boxes, conf[:,None]])
                preds_list.append(_preds)
        if preds_list:
            preds = np.vstack(preds_list)
            return preds[preds[:, -1] > conf_thres]
        else:
            return np.empty((0, 10))
    
    def checkROI(self, roi):
        if roi is None or roi.size == 0 or min(roi.shape[:2]) == 0:
            return False 
        roi   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi   = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)[1]  # More efficient
        roi   = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        white = np.sum(roi == 255)
        total = roi.size  # To
        black = total - white
        ratio = white/black if black!=0 else white   
        return ratio > 0.9
    
    def pinPost_process(self, img, pinPreds, numpins=21, pinHeight=400, pinWidth=24):
        if pinPreds.shape[0] <= numpins:
            return pinPreds
        pinPreds = pinPreds[np.argsort(pinPreds[:, 6])]
        # filter small preds (preds with less height and width may be predicted as extra unnecessary pred)
        hs = pinPreds[:,3] - pinPreds[:,7]
        ws = pinPreds[:,4] - pinPreds[:,6]
        valid_mask = (hs > pinHeight) & (ws > pinWidth)
        pinPreds   = pinPreds[valid_mask] 
        # merge overlapping preds
        if pinPreds.shape[0]>numpins:
            topL_xs    = pinPreds[:,6]
            separation = np.diff(topL_xs)
            merge_idx  = np.where(separation < 4)[0]
            pinPreds[merge_idx] = (pinPreds[merge_idx] + pinPreds[merge_idx + 1]) / 2
            pinPreds = np.delete(pinPreds, merge_idx + 1, axis=0)

        if pinPreds.shape[0]>numpins:
            # Check overlapping pins and keep the larger ones
            areas = np.zeros(pinPreds.shape[0])
            for i in range(pinPreds.shape[0]):
                # Calculate area of each pin using coordinates
                box = pinPreds[i].reshape(4,2)
                area = cv2.contourArea(np.int32(box))
                areas[i] = area

            # Initialize mask for pins to keep
            keep = np.ones(pinPreds.shape[0], dtype=bool)

            # Compare each pair of pins
            for i in range(pinPreds.shape[0]):
                if not keep[i]:
                    continue
                box1 = pinPreds[i].reshape(4,2)
                for j in range(i + 1, pinPreds.shape[0]):
                    if not keep[j]:
                        continue
                    box2 = pinPreds[j].reshape(4,2)
                    
                    # Check intersection using rotated rectangle intersection
                    intersection = cv2.rotatedRectangleIntersection(
                        cv2.minAreaRect(np.int32(box1)),
                        cv2.minAreaRect(np.int32(box2))
                    )[0]
                    
                    # If boxes overlap, keep the larger one
                    if intersection is not None and intersection >= 1:
                        if areas[i] >= areas[j]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break

            pinPreds = pinPreds[keep]
        # filter unnecessary preds in between pins
        if pinPreds.shape[0] > numpins:
            topLxs = pinPreds[1:, 6]
            topRxs = pinPreds[:-1, 4]
            separation = topLxs - topRxs
            idx = np.where(separation < 15)[0]
            if idx.size > 0:
                rois_to_check = [
                            (pinPreds[i], img[int(pinPreds[i][7]):int(pinPreds[i][3]), int(pinPreds[i][6]):int(pinPreds[i][2])])
                            for i in idx
                                        ] + [
                            (pinPreds[i + 1], img[int(pinPreds[i + 1][7]):int(pinPreds[i + 1][3]),
                            int(pinPreds[i + 1][6]):int(pinPreds[i + 1][2])])
                            for i in idx
                            ]
                roi_states = np.array([self.checkROI(roi) for _, roi in rois_to_check])

                nonPinIdxs = np.ones(pinPreds.shape[0], dtype=bool)
                idx_pairs = np.concatenate([idx, idx + 1])
                nonPinIdxs[idx_pairs] = roi_states

                pinPreds = pinPreds[nonPinIdxs]               
        return pinPreds
    
    def processPins(self, img, preds, surface, adjustment=0, num_pins=21, roi=[]):
        '''Post-processes predicted YOLO ROIs for pins and separates upper/lower pins in Front view.'''
        
        def pinSeparation(_preds, numPins=num_pins):
            '''Separate upper and lower pins in front views'''
            lowerPin_thres = np.max(_preds[:, [5, 7]])
            pin1 = preds[np.all(preds[:, [1, 3, 5, 7]] < lowerPin_thres, axis=1)]
            pin1 = pin1[pin1[:, 0].argsort()]

            if pin1.shape[0] > numPins:
                pin1 = self.pinPost_process(img, pin1, numPins)

            # Top y alignment
            topThres = np.average(pin1[:, [5, 7]])
            pin1[:, [5, 7]] = np.clip(pin1[:, [5, 7]], topThres, topThres + 10)

            # Bottom y alignment
            thres = np.max(pin1[:, [1, 3]]) - 5
            pin1[:, [1, 3]] = np.clip(pin1[:, [1, 3]], thres, thres)
            return pin1

        pin1 = None
        front_surfaces = ['Front11', 'Front12', 'Front21', 'Front22']
        top_group1 = ['Top11', 'Top21']
        top_group2 = ['Top12', 'Top22']

        if surface in front_surfaces:
            pin1 = pinSeparation(preds)
            pin1[:, [0, 6]] -= adjustment
            pin1[:, [2, 4]] += adjustment

        elif surface in top_group1 or surface in top_group2:
            pin1 = preds

            if pin1.shape[0] != num_pins:
                pin1 = self.pinPost_process(img, pin1, num_pins)

            # Top y alignment
            topThres = np.average(pin1[:, [5, 7]])
            pin1[:, [5, 7]] = np.clip(pin1[:, [5, 7]], topThres, topThres)

            # Bottom y alignment
            offset = 10 if surface in top_group1 else 2
            bottomThres = np.average(pin1[:, [1, 3]]) + offset
            pin1[:, [1, 3]] = np.clip(pin1[:, [1, 3]], bottomThres, bottomThres)

            # Horizontal adjustments
            pin1[:, [0, 6]] -= adjustment
            pin1[:, [2, 4]] += adjustment

        return pin1

    def generateMask(self, img, obbBoxes, itrs=1, callfrom=''):
        # s = time.time()
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=bool)

        for box in obbBoxes:                     
            x1, y1 = max(0, int(box[6]-3)), max(0, int(box[7]))
            x2, y2 = min(width, int(box[2]+3)), min(height, int(box[3]))
            
            patch = img[y1:y2,x1:x2]
            patch = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)[1]
            # patch = cv2.morphologyEx(patch, cv2.MORPH_DILATE, self.kernel, iterations=1)

            h, w  = patch.shape

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=4)

            # post process to remove very small connected components and extending bigger components from top and bottom if necessary
            if num_labels>0:
                for i in range(1, num_labels):  # Skip background (label 0)
                    if stats[i, cv2.CC_STAT_AREA] < 1000:
                        continue
                    x, y, _w, _h, area = stats[i]
                    _x1 = int(x+3)
                    _x2 = int(x+_w-3)
                    _y1 = int(y)
                    # _y2 = int(y+_h)
                    # if _y1>5:
                    #     _y1 = 2   
                    #     # y1 = 5
                    # if _y2<((h/2)+10):
                    _y2 = h-3
                    
                    patch[_y1:_y2, _x1:_x2] = 255    

            # post processing for patch Bottom
            y = int(0.7 * h)
            x_coords = np.where(patch[-20, :] > 0)[0]

            left_x = np.min(x_coords) if x_coords.size > 0 else _x1 + 2
            right_x = np.max(x_coords) if x_coords.size > 0 else _x2 -2
            patch[y:h-3, left_x:right_x] = 255

            # patch[y:height-1, left_x:right_x] = 255
            
            patch[[0,-1],:] = 0
            patch[:,[0,-1]] = 0
            patch = cv2.morphologyEx(patch, cv2.MORPH_DILATE, self._kernel, iterations=itrs+2)
            contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            smoothed_mask = np.zeros_like(patch)

            for cnt in contours:
                if cv2.contourArea(cnt) < 1000:
                    continue
                approx = cv2.approxPolyDP(cnt, epsilon=5, closed=True)
                cv2.drawContours(smoothed_mask, [approx], -1, 255, thickness=cv2.FILLED)
            if 'Top' in self.surface:
                # Create a copy of smoothed mask
                patch = smoothed_mask.copy()
                h = patch.shape[0]
                # Apply dilation to top 50 rows
                patch[:70] = cv2.morphologyEx(smoothed_mask[:70], cv2.MORPH_DILATE, self._kernel, iterations=1)
                # No erosion for first 50 rows
                patch[70:h-150] = cv2.morphologyEx(smoothed_mask[70:h-150], cv2.MORPH_ERODE, self._kernel, iterations=itrs)
                # More erosion for bottom rows
                patch[h-150:] = cv2.morphologyEx(smoothed_mask[h-150:], cv2.MORPH_ERODE, self._kernel, iterations=itrs+2)
            elif 'Front' in self.surface:
                if self.surface in ['Front-pin-2nd_auto_0', 'Front-pin_auto_1', 'Front-pin-2nd_auto_1']:
                    patch = cv2.morphologyEx(smoothed_mask, cv2.MORPH_ERODE, self._kernel, iterations=itrs)   
                elif self.surface in ['Front-pin_auto_0']:
                    patch = cv2.morphologyEx(smoothed_mask, cv2.MORPH_ERODE, self._kernel, iterations=itrs+1)   
            mask[y1:y2, x1:x2] = patch>0
            mask[y1-30:y1+5, x1:x2] = True
            bottom_offset = y2+20 if y2+20<height else height-1
            mask[y2-5:bottom_offset, x1:x2] = True
        # print('Generate mask method time: ', time.time()-s)
        return mask
    
    def segment(self, img, obbBoxes, roi='Bur'):
        # s = time.time()
        roiBox = self.ROIbox(obbBoxes, padding=(60,0))
        # adjustment for bur ROI
        roiBox[1] += 10
        roiBox[3] -= 10

        roiImg  = np.zeros_like(img, dtype=np.uint8)
        x1, y1, x2, y2 = roiBox
        roiImg[y1:y2, x1:x2] = img[y1:y2, x1:x2]
        mask = self.generateMask(img, obbBoxes, itrs=1, callfrom='segment_burr')
        mask = ~mask
        roiImg = cv2.bitwise_and(roiImg, roiImg, mask=mask.astype(np.uint8))
        # print('Segment method time (Burr): ', time.time()-s)      
        return roiImg, roiBox    

    def ROIbox(self, obbPreds, width=1920, height=1280, padding=(30,10)):
        # s = time.time()

        # Extract x and y coordinates
        coords = obbPreds[:, [0, 2, 4, 6, 1, 3, 5, 7]]
        x_coords, y_coords = coords[:, :4], coords[:, 4:]

        # Compute min and max with padding
        x_min, x_max = np.min(x_coords, axis=1) - padding[0], np.max(x_coords, axis=1) + padding[0]
        y_min, y_max = np.min(y_coords, axis=1) - padding[1], np.max(y_coords, axis=1) + padding[1]

        # Clip values within bounds
        x_min, x_max = np.clip([x_min, x_max], 0, width - 1).astype(int)
        y_min, y_max = np.clip([y_min, y_max], 0, height - 1).astype(int)

        # print(f'Method ROIbox process time: {time.time() - s:.6f} sec')

        return [min(x_min), min(y_min), max(x_max), max(y_max)]
    
    def filterPreds(self,predictions, roi):
        x_min, y_min, x_max, y_max = roi

        # Reshape predictions to (N, 4, 2) for easy indexing of points
        points = predictions.reshape(-1, 4, 2)

        # Check if all points lie within the ROI
        inside_x = (points[:, :, 0] >= x_min) & (points[:, :, 0] <= x_max)  # x-coordinates within ROI
        inside_y = (points[:, :, 1] >= y_min) & (points[:, :, 1] <= y_max)  # y-coordinates within ROI
        inside_roi = np.all(inside_x & inside_y, axis=1)  # Check all corners are inside

        # Filter predictions based on the ROI condition
        preds = predictions[inside_roi]

        return preds    
    
    def burrROI(self, img, pin):
        return self.segment(img, pin, 'Bur')

    def drawBox(self, img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1, text=''):
        # s = time.time()
        if obb:
            for box in boxes:
                if box.shape[0]==10:
                    _box = np.array([[int(box[1]), int(box[2])],
                                    [int(box[3]), int(box[4])],
                                    [int(box[5]), int(box[6])],
                                    [int(box[7]), int(box[8])]])
                elif box.shape[0]==8:
                    _box = np.array([[int(box[0]), int(box[1])],
                                    [int(box[2]), int(box[3])],
                                    [int(box[4]), int(box[5])],
                                    [int(box[6]), int(box[7])]])    
                _box = _box.reshape(-1,1,2)
                img  = cv2.polylines(img, [_box], isClosed=True, color=color, thickness=thickness)
        else:
            if boxType=='xywh':
                for box in boxes:
                    x1 = int(box[1] - box[3]/2)
                    x2 = int(box[1] + box[3]/2)
                    y1 = int(box[2] - box[4]/2)
                    y2 = int(box[2] + box[4]/2)

                    img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    text = f'{box[0]}:'
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                    label_bg_top_left = (x1, y1 - h - 5)
                    label_bg_bottom_right = (x1 + w, y1)
                    cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)

            elif boxType=='xyxy':   # TODO: implement the box rendering for xyxy box coordinates
                for box in boxes:
                    x1 = int(box[0])
                    x2 = int(box[2])
                    y1 = int(box[1])
                    y2 = int(box[3])

                    img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    text = f'{text}:'
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                    label_bg_top_left = (x1, y1 - h - 5)
                    label_bg_bottom_right = (x1 + w, y1)
                    cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)
        # print(f'Method drawBox process time: {time.time()-s}')
        return img    
    
    def getCrop_offsets(self, pins=[], roiBox=[]):
        pinPoses = []
        x1, y1, w, h = int(roiBox[0]), int(roiBox[1]), int(roiBox[2]), int(roiBox[3])
        roiH = h-y1
        for pin in pins:
            x2 = int(((pin[0]+pin[2])/2)-5)
            y2 = int(y1 + (roiH/2))
            pinPoses.append([x1, y1, x2,y2])
            pinPoses.append([x1, y2, x2,h])
            x1 = x2
        # pinPoses.append([x1, int(y1/2), w, h])
        y2 = int(y1 + (roiH/2))
        pinPoses.append([x1, y1, w, y2])
        pinPoses.append([x1, y2, w, h])
        return pinPoses

    def getPatch(self, img, pinPoses):
        # patch = img[pinPoses[1]:pinPoses[3],pinPoses[0]:pinPoses[2]]
        return img[pinPoses[1]:pinPoses[3],pinPoses[0]:pinPoses[2]]
    
    def patchify(self, img=[], pinPreds=None, roi=[]):
        patches, patch_positions = [], []
        if len(pinPreds)>=20:
            pinPoses = self.getCrop_offsets([pinPreds[3], pinPreds[7], pinPreds[11], pinPreds[15], pinPreds[18]], roi)
            for pos in pinPoses:
                patch = self.getPatch(img,pos)
                patches.append(cv2.resize(patch, (640,640), cv2.INTER_NEAREST))
                patch_positions.append(pos)
            # cv2.destroyAllWindows()        
        elif (len(pinPreds))<=19:            
            pinPoses = self.getCrop_offsets([pinPreds[2], pinPreds[6], pinPreds[10], pinPreds[14], pinPreds[17]], roi)
            for pos in pinPoses:
                patch = self.getPatch(img,pos)
                patches.append(cv2.resize(patch, (640,640), cv2.INTER_NEAREST))
                patch_positions.append(pos)           
        # cv2.destroyAllWindows()
        return {"patches": patches, "patch_positions": np.array(patch_positions)}
    
    def getNameAndRoi(self, realName, meanROI):
        x1, y1, x2, y2 = meanROI

        # roi_params = {
        #     "Top-pin-2nd_auto_0":  (-830, 15, 55, 920, 1890, 1910, -15, 550, 650, 8, 1275, 1290), # done
        #     "Top-pin-2nd_auto_1":  (-830, 30, 90, 860, 1890, 1915, -20, 550, 650, 8, 1275, 1290), # done
        #     "Top-pin_auto_0":  (-810, 0, 25, 870, 1790, 1870, -15, 420, 450, 35, 1260, 1275), # done
        #     "Top-pin_auto_1":  (-830, 0, 15, 890, 1810, 1860, -40, 430, 460, 45, 1260, 1275),  # done
        #     "Front-pin-2nd_auto_0": (-830, 25, 90, 910, 1900, 1915, -320, -3, 0, 150, 880, 910), # done
        #     "Front-pin-2nd_auto_1": (-830, 40, 90, 910, 1901, 1919, -320, -1, 3, 150, 880, 910), # done
        #     "Front-pin_auto_0": (-830, 1, 25, 910, 1880, 1910, -350, 40, 95, 150, 930, 990),   # done
        #     "Front-pin_auto_1": (-830, 1, 25, 910, 1870, 1900, -330, 40, 95, 150, 930, 990),   # done
        # }

        roi_params = {
            "Top-pin-2nd_auto_0":  (-830, 15, 60, 920, 1880, 1900, -15, 550, 650, 8, 1282, 1290), # done
            "Top-pin-2nd_auto_1":  (-830, 15, 100, 860, 1880, 1900, -20, 550, 650, 8, 1282, 1290), # done
            "Top-pin_auto_0":  (-810, 20, 70, 870, 1820, 1860, -15, 420, 450, 35, 1265, 1280), # done
            "Top-pin_auto_1":  (-830, 20, 70, 890, 1810, 1870, -40, 430, 460, 45, 1265, 1280),  # done
            "Front-pin-2nd_auto_0": (-830, 10, 130, 910, 1900, 1918, -320, -3, 0, 150, 880, 910), # done
            "Front-pin-2nd_auto_1": (-830, 10, 130, 910, 1901, 1919, -320, -1, 3, 150, 880, 910), # done
            "Front-pin_auto_0": (-830, 1, 30, 910, 1880, 1910, -350, 40, 95, 150, 930, 990),   # done
            "Front-pin_auto_1": (-830, 1, 30, 910, 1870, 1900, -330, 40, 95, 150, 930, 990),   # done
            "Front-pin_auto_0_ModifiedExposure": (-830, 1, 30, 910, 1880, 1910, -350, 40, 95, 150, 930, 990),   # done
            "Front-pin_auto_1_ModifiedExposure": (-830, 1, 30, 910, 1870, 1900, -330, 40, 95, 150, 930, 990),   # done
            "Top-pin_auto_0_ModifiedExposure": (-810, 20, 70, 870, 1820, 1860, -15, 420, 450, 35, 1265, 1280), # done
            "Top-pin_auto_1_ModifiedExposure": (-830, 20, 70, 890, 1810, 1870, -40, 430, 460, 45, 1265, 1280),  # done
        }

        for key, (dx1, min_x1, max_x1, dx2, min_x2, max_x2, dy1, min_y1, max_y1, dy2, min_y2, max_y2) in roi_params.items():
            if key==realName:
                _x1 = min(max(x1 + dx1, min_x1), max_x1)
                _x2 = min(max(x2 + dx2, min_x2), max_x2)
                _y1 = min(max(y1 + dy1, min_y1), max_y1)
                _y2 = min(max(y2 + dy2, min_y2), max_y2)
                roi = [_x1, _y1, _x2, _y2]
                break
        else:
            return (None, None, None, None)  # If no match is found

        mapping = {
            "Front-pin-2nd_auto_0": ("Front12", roi, -1, 21),
            "Front-pin-2nd_auto_1": ("Front22", roi, -1, 19),
            "Front-pin_auto_0": ("Front21", roi, -1, 19),
            "Front-pin_auto_1": ("Front11", roi, -1, 21),
            "Top-pin-2nd_auto_0": ("Top12", roi, -3, 21),
            "Top-pin-2nd_auto_1": ("Top22", roi, -3, 19),
            "Top-pin_auto_0": ("Top21", roi, -3, 19),
            "Top-pin_auto_1": ("Top11", roi, -3, 21),
            "Front-pin_auto_0_ModifiedExposure": ("Front21", roi, -1, 19),
            "Front-pin_auto_1_ModifiedExposure": ("Front11", roi, -1, 21),
            "Top-pin_auto_0_ModifiedExposure": ("Top21", roi, -3, 19),
            "Top-pin_auto_1_ModifiedExposure": ("Top11", roi, -3, 21),
        }

        return mapping.get(realName, (None, None, None, None))    
    
    def getBoxes(self, roi):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        h,w,_  = roi.shape

        patch = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        patch = cv2.inRange(patch, 36, 255)
        patch = cv2.erode(patch, kernel, iterations=1)
        patch[0:150,0:w] = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(patch, connectivity=4)
        
        # post process to remove very small connected components and extending bigger components from top and bottom if necessary
        if num_labels>0:
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] < 2500:
                    patch[labels == i] = 0
                else:
                    x, y, _w, _h, area = stats[i]
                    x1 = int(x+8)
                    x2 = int(x+_w-1)
                    y1 = int(y+2)
                    y2 = int(y+_h-1)
                    if y1>50:
                        patch[8:y1+5,x1:x2]=255
                        # y1 = 5
                    if y2<((h/2)+10):
                        patch[y2:h-1, x1:x2]=255
                        # y2 = h-1
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        if len(contours)>0:
            for contour in contours:
                if cv2.contourArea(contour) < 2500:
                    continue
                rect = cv2.minAreaRect(contour)

                box = cv2.boxPoints(rect)

                box = box[np.argsort(box[:,1])]
                if box[0,0]>box[1,0]: box[[0,1]] = box[[1,0]]
                if box[2,0]<box[3,0]: box[[2,3]] = box[[3,2]]
                box = box[::-1].flatten()
                # if box[5]>=8:
                #     box[5]=1
                # if box[1]<=h-10:
                #     box[1] = h-1                
                boxes.append(box)
        return np.vstack(boxes) if boxes else np.empty((0, 4))
    
    def recoverPinBoxes(self, img, preds=[], roi=[]):
        _boxes = []
        sort_idx = np.argsort(preds[:, 6])
        preds = preds[sort_idx]
        # check if the pins are missing at the start or end and recover
        startPred = preds[0].astype(int)
        endPred   = preds[-1].astype(int)
        if startPred[6]-roi[0] > 100:
            patch = img[startPred[7]:startPred[1], roi[0]+20:startPred[0]-5]
            boxes = self.getBoxes(patch)
            if boxes.shape[0]>0:
                boxes[:,::2] += roi[0]+20    
                # boxes[:,1::2] += roi[1]+20
                boxes[:,[1, 3, 5,7]] = startPred[1],startPred[3],startPred[5],startPred[7]
                _boxes.append(boxes)
        if roi[2]-endPred[4] > 120:
            patch = img[endPred[5]:endPred[1], endPred[4]+5:roi[2]-20]
            boxes = self.getBoxes(patch)
            if boxes.shape[0]>0:
                boxes[:,::2] += endPred[4]+5    
                # boxes[:,1::2] += patchTop[1]
                boxes[:,[1, 3, 5,7]] = endPred[1], endPred[3], endPred[5], endPred[7]
                _boxes.append(boxes)        
        topLxs = preds[:,6]
        separation = np.diff(topLxs)
        idx = np.where(separation>115)[0]
        # if numpins ==19:
        #     idx = idx[idx<14]
        for _idx in idx:
            pred1, pred2 = preds[_idx], preds[_idx + 1]
            patchTop, patchBottom = (int(pred1[4] + 4), int(pred1[5])), (int(pred2[0] - 4), int(pred2[1]))
            patch = img[patchTop[1]:patchBottom[1], patchTop[0]:patchBottom[0]]
            boxes = self.getBoxes(patch)
            if boxes.shape[0]>0:
                boxes[:,::2] += patchTop[0]    
                boxes[:,1::2] += patchTop[1]
                boxes[:,[1, 3, 5,7]] = startPred[1],startPred[3],startPred[5],startPred[7]
                _boxes.append(boxes)
        return np.concatenate(_boxes, axis=0) if _boxes else np.empty((0, 8))
    
    def check_overlap(self, boxes):
        overlaps = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                # Convert OBB coordinates to point format
                box1_pts = boxes[i].reshape(4,2)
                box2_pts = boxes[j].reshape(4,2)
                
                # Create contours
                contour1 = np.int32(box1_pts)
                contour2 = np.int32(box2_pts)
                
                # Check intersection using contour intersection
                intersection = cv2.rotatedRectangleIntersection(
                    cv2.minAreaRect(contour1),
                    cv2.minAreaRect(contour2)
                )[0]
                
                if intersection is not None and intersection >= 1:
                    overlaps += 1
        
        return overlaps
    
    def process(self, image: np.ndarray, info={}): #TODO 7: You will receive image and the detection results from previous model (if this model is not the first one)

        try:
            # logger.info(f"Surface Name: {info['Input']}")
            # logger.info(f"Image Acquisition Time: {datetime.datetime.now()}")    
            # fst = time.time()
            
            # Check if the input image is valid
            if image is None:
                raise ValueError("Input image is empty or None.")
            
            # logger.info(f"[Get Update param] Time taken: {(time.time() - time.time()) * 1000:.4f} milliseconds")
            
            # start_time = time.time()   

            # cloning time
            st = time.time()
            img = np.copy(image)
            # logger.info(f"[Cloning image] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")

            # st = time.time()
            # Convert to CuPy array
            # img_gpu = cp.asarray(img)
            img = cv2.resize(img, (1920, 1280), interpolation=cv2.INTER_LINEAR)
            realName = info['Input']
            logger.debug(f"[Info Recieved to ROI Model]: {info}")
            logger.debug(f"[Surface Name]: {realName}")
            self.surface = realName
            logger.info(f"[clonning] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
            # print(realName)

            try:
                # Run inference
                if realName in ['Front-pin-2nd_auto_0', 'Top-pin-2nd_auto_1']:
                    with self.model_lock1:
                        st = time.time()
                        results = self.model_1.predict(img, conf=self.conf_thres, device=self.device, verbose=False)
                        self.logger.info(f"[ROI Model_1 Inference] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                elif realName in ['Front-pin-2nd_auto_1', 'Top-pin-2nd_auto_0']:
                    with self.model_lock2:
                        st = time.time()
                        results = self.model_2.predict(img, conf=self.conf_thres, device=self.device, verbose=False)
                        self.logger.info(f"[ROI Model_2 Inference] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                elif realName in ['Front-pin_auto_0', 'Top-pin_auto_1', 'Front-pin_auto_0_ModifiedExposure', 'Top-pin_auto_1_ModifiedExposure']:
                    with self.model_lock3:
                        st = time.time()
                        results = self.model_3.predict(img, conf=self.conf_thres, device=self.device, verbose=False)
                        self.logger.info(f"[ROI Model_3 Inference] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                elif realName in ['Front-pin_auto_1', 'Top-pin_auto_0', 'Front-pin_auto_1_ModifiedExposure', 'Top-pin_auto_0_ModifiedExposure']:
                    with self.model_lock4:
                        st = time.time()
                        results = self.model_4.predict(img, conf=self.conf_thres, device=self.device, verbose=False)
                        self.logger.info(f"[ROI Model_4 Inference] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")                        
                
                # st = time.time()
                preds   = self.processOBB(results, self.conf_thres)
                # logger.info(f"[Prediction Post-process] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                # print('Model Inference Time: ', time.time()-_s)
                
                
                if preds.shape[0]>10:
                    # st = time.time()
                    # print(preds.shape[0])
                    preds = preds[:,1:9]
                    preds = np.clip(preds, [0, 0, 0, 0, 0, 0, 0, 0], [self.w-1, self.h-1]*4)

                    _roi = (np.add.reduce(preds[:, [6, 7, 2, 3]], axis=0) / preds.shape[0]).astype(int)
                    name, roi, adj, numPins = self.getNameAndRoi(realName, _roi)
                    # logger.info(f"[upper pin extraction preprocess] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                    
                    # For ROI  verification, uncomment below lines to verify and adjust the roi box for filtering unwanted box outside ROI
                    # _img = copy.deepcopy(img)
                    # _img = self.drawBox(_img, [_roi], boxType='xyxy')
                    # _img = self.drawBox(_img, [roi], boxType='xyxy')
                    # cv2.imwrite('img.jpg', _img)

                    # st = time.time()
                    if roi is not None:
                        _preds = self.filterPreds(preds, roi)
                        if _preds.shape[0]>0:
                            preds = _preds
                        elif _preds.shape[0]==0:
                            self.logger.info(f"\t\t{(Fore.RED)}{(Style.BRIGHT)}{(emoji.emojize(':warning:'))} {' WARNING'} All predictions removed with PinROI based filtering, Please Check input image and adjust PinROI {Style.RESET_ALL}")
                            preds = preds
                    # logger.info(f"[FP filtering] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")        
                    
                    # st = time.time()
                    pin1 = self.processPins(img=img, preds=preds, surface=name, adjustment=adj, num_pins=numPins, roi=roi)
                    # logger.info(f"[upper pin post-process] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                    # manual tests for missing pin recovery
                    # pin1 = np.delete(pin1, [10,12], axis=0)

                    # st = time.time()
                    if pin1.shape[0]<numPins:
                        recBoxes = self.recoverPinBoxes(img, pin1, roi)
                        if recBoxes.any(): pin1 = np.concatenate((pin1,recBoxes), axis=0)
                    # logger.info(f"[Pin recovery] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")        
                    # Check for overlapping predictions

                    num_overlaps = self.check_overlap(pin1)
                    if pin1.shape[0]>numPins and num_overlaps > 0:
                        self.logger.warning(f"Found {num_overlaps} overlapping pin predictions")
                        det_results = []
                    # For pin prediction verification
                    # uncomment below lines to verify all the pins are predicted after prediction and post-process steps
                    # _img = copy.deepcopy(img)
                    # _img = self.drawBox(_img, pin1, obb=True)
                    # cv2.imwrite('img.jpg', _img)
                    # _s = time.time()
                    else:
                        sorted_indices = np.argsort(pin1[:, 6])
                        pin1       = pin1[sorted_indices]
                        # st = time.time()    
                        burImg, burROI_box  = self.burrROI(img=img, pin=pin1)
                        # reduction in roi for burr detection
                        if 'Front' in realName:
                            burROI_box[1] += 5
                            burROI_box[3] -= 50
                        elif 'Top-pin_auto' in realName:
                            burROI_box[1] += 5
                            # burROI_box[3] -= 5

                            burROI_box[0] += 15
                            burROI_box[2] -= 25
                        elif 'Top-pin-2nd' in realName:
                            burROI_box[1] += 5
                            # burROI_box[3] -= 1

                            burROI_box[0] += 20
                            burROI_box[2] -= 30        
                        # print('ROI extraction time: ', time.time()-_s) 

                        det_results = {
                            'Surface_names': [realName, name],
                            'original_img' : img,
                            'upperPinsObb_preds': pin1,
                            'burROI_img': burImg,
                            'burROI_box': burROI_box
                        }
                            # logger.info(f"[BurrROI extraction] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                    
                    # logger.info(f"[Post process] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
                    # logger.success(f"[Overall Process] Time taken: {(time.time() - fst) * 1000:.4f} milliseconds")
                    # logger.info(f"Result Ready for {info['Input']}: False at {datetime.datetime.now()}")
                    # logger.info(" ")
                    # logger.info("------------------------------------------------------------------------------")
                    # logger.info("")

                    # img = self.drawBox(img, pin1, obb=True, color=(0,0,255))
                    # img = self.drawBox(img,[roi], boxType='xyxy')
                    # Pin_img = self.drawBox(Pin_img, np.array([pinROI_box[0]]), boxType='xyxy', thickness=2, text=f'{realName}; {name}')
                    
                    # cv2.imwrite('img.jpg', img)
                    # cv2.imwrite('img.jpg', img)
                    # cv2.imwrite('burImg.jpg', burImg)
                    # cv2.imwrite('pinImg.jpg',Pin_img)
                    # time.sleep(1)
                elif preds.shape[0]<=10:
                    print(f"\t\t{(Fore.RED)}{(Style.BRIGHT)}{(emoji.emojize(':warning:'))} {' WARNING'} Nothing Detected, Please Check input image {Style.RESET_ALL}")
                    det_results=[]
                    # logger.info(f"[Post process] Time taken in Fail case: {(time.time() - st) * 1000:.4f} milliseconds")
                    # logger.success(f"[Overall Process] Time taken: {(time.time() - fst) * 1000:.4f} milliseconds")
                    # logger.info(f"Result Ready for {info['Input']}: False at {datetime.datetime.now()}")
                    # logger.info(" ")
                    # logger.info("------------------------------------------------------------------------------")
                    # logger.info("")
            except Exception as e:
                self.logger.exception(f"Error in model inference: {e}")
                det_results=[]
            
            # print("pinROI Model Processing Time: ", time.time()-start_time)
            # uncomment below line for testing as standalone and comment the below it
            # return det_results, pin1, img
            return det_results
        except Exception as e:
            # print(f"Error in pinROI model: {e}")
            self.logger.exception(f"Error in pinROI model: {e}")
            return []
if __name__ == "__main__":
    import tqdm
    import argparse
    import os

    def arguments():
        parser = argparse.ArgumentParser(description='Demo')
        parser.add_argument('--dataPath', type=str, default='',
                            help='Provide the path of data.txt file')
        parser.add_argument('--roiModel', type=str, default=None,
                            help='Provide model.pt path')
        parser.add_argument('--roiConf', type=float, default=0.5,
                            help='Choose a confidence threshold between 0 and 1')
        parser.add_argument('--device', type=int, default=0,
                            help='Choose a gpu device')
        parser.add_argument('--savePath', type=str, default='',
                            help='Provide a path to save the resultant image')
        args = parser.parse_args()
        return args
    
    args = arguments()
    # Load the model
    model = MODEL(model_path=args.roiModel, confidence_threshold=0.2, warmup_runs=3)

    dataPath = args.dataPath
    savePath = args.savePath

    with open(dataPath, 'r') as f:
        imgPaths = f.readlines()
    
    idx = 0
    # Load an image
    for imagePath in tqdm.tqdm(imgPaths):
        print(imagePath)
        # imagePath   = "/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/new_data_factory_bldng/tml_burr_from_ket_20250408-Top-pin-2nd_auto_1.bmp"
        imagePath   = imagePath.strip()
        surfaceName = imagePath.split('/')[-2]
        # surfaceName = 'Top-pin-2nd_auto_1'
        info={'Input':surfaceName}
        image = cv2.imread(imagePath)

        # Run inference
        rslts = model.process(image, info=info)

        pinPreds    = rslts['upperPinsObb_preds']
        burImg      = rslts['burROI_img']
        burROI_box  = rslts['burROI_box']
        # Burmask     = rslts['burROI_mask']
        
        sorted_indices = np.argsort(pinPreds[:, 6])
        pinPreds = pinPreds[sorted_indices]
        # print(len(pinPreds))

        # adjust roi to igonre black coated region
        # if 'Front' in imagePath:
        #     burROI_box[1] += 10
        #     burROI_box[3] -= 150
        # elif 'Top' in imagePath:
        #     burROI_box[1] += 5
        #     burROI_box[3] -= 10

        # Burmask = (Burmask*255).astype(np.uint8)    

        # preprocess brr roi image
        _burImg = cv2.cvtColor(burImg, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(_burImg)
        # v       = cv2.bilateralFilter(v, 5, 0.05, 5)

        if 'Top' in imagePath:
            v_gamma = np.clip(255*(v/255)**0.5, 0, 255)
        elif 'Front' in imagePath:
            v_gamma = np.clip(255*(v/255)**0.5, 0, 255)        
        # v_gamma_bright = np.clip(v_gamma*1.6, 0, 255).astype(np.uint8)
        v_gamma[v_gamma<10] = 0
        v_gamma_bright = np.clip(v_gamma*1.3, 0, 255).astype(np.uint8)
        # v_gamma_bright = cv2.bilateralFilter(v_gamma_bright, 7, 0.1, 5)
        _burImg_filterd_th_gamma = cv2.merge([h, s, v_gamma_bright])
        _burImg_filterd_th_gamma = cv2.cvtColor(_burImg_filterd_th_gamma, cv2.COLOR_HSV2BGR)

        # cv2.imshow('',_burImg_filterd_th_gamma)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        # patchify
        patchedImg = model.patchify(_burImg_filterd_th_gamma, pinPreds=pinPreds, roi=burROI_box)
        patches, patchPosz, patchmasks = patchedImg.get('patches'), patchedImg.get('patch_positions'), patchedImg.get('patch_masks')

        # write patched images
        for i, patch in enumerate(patches):
            # patchMask = patchmasks[i]
            imgSavePath = os.path.join(savePath, 'images')
            img_name    = os.path.join(imgSavePath, f'img_{idx}.png')
            # maskSavePath = os.path.join(savePath, 'masks')
            # mask_name    = os.path.join(maskSavePath, f'burImg_{idx}.png')

            idx += 1

            cv2.imwrite(img_name, patch)