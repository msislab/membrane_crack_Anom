import cv2
import numpy as np
import copy
from ultralytics import YOLO
import glob
import os


class ROI_model:
    '''Class to extract different pin ROIs'''
    def __init__(self, modelPath: str, warmup_runs=1):

        self.model = YOLO(modelPath)

        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(warmup_runs):
            _ = self.model.predict(source=dummy_image, conf=0.6, verbose=False)

    def visPreds(self, img=None, preds=None, obb=False, boxType='xywh', color=(0,0,255)):
        if obb:
            self.drawBox(img, preds, obb=obb, color=color, thickness=2)
        else:
            self.drawBox(img, preds, boxType=boxType, color=color)

    def drawBox(self, img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1):
        if obb:
            # print('visualizing obb preds')
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
                # text = f"{box[0]};{box[-1]}"
                # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                # label_bg_top_left = (int(box[1]), int(box[2]) - h - 5)
                # label_bg_bottom_right = (int(box[1]) + w, int(box[2]))
                # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                # cv2.putText(img, text, (box[1]), (int(box[2]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)
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
                    x1 = int(box[1])
                    x2 = int(box[3])
                    y1 = int(box[2])
                    y2 = int(box[4])

                    img  = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    text = f'{box[0]}:'
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                    label_bg_top_left = (x1, y1 - h - 5)
                    label_bg_bottom_right = (x1 + w, y1)
                    cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)      
        cv2.imshow('', img)
        cv2.waitKey()
        cv2.destroyAllWindows()        
    
    def preProcess(self, img, surface, blob_area=120):
        # hard coded parameters (change accordingly with the change in image view)
        if surface=='Front1':
            roi_1   = [110, 165, 1800, 705]
            roi_2   = [110, 740, 1800, 905]
        elif surface=='Front2':
            roi_1 = [100, 150, 1815, 725]
            roi_2 = [100, 735, 1815, 910]
        elif surface=='Top1':
            roi_1 = [100, 150, 1815, 725]
            roi_2 = [100, 735, 1815, 910]
        elif surface=='Top2':
            roi_1 = [100, 150, 1815, 725]
            roi_2 = [100, 735, 1815, 910]
        # preprocess
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(grayImg, dtype=np.uint8)

        mask[roi_1[1]:roi_1[3], roi_1[0]:roi_1[2]] = 255
        mask[roi_2[1]:roi_2[3], roi_2[0]:roi_2[2]] = 255

        grayImg = cv2.bitwise_and(grayImg, grayImg, mask=mask)

        outside_roi = cv2.bitwise_not(mask)  # Invert mask
        grayImg[outside_roi > 0] = 0  # Set outside pixels to 0

        oriImg = copy.deepcopy(grayImg)

        grayImg = cv2.medianBlur(grayImg, 5)
        # grayImg = cv2.GaussianBlur(grayImg, (5, 5), 0)

        # grayImg[grayImg<20] = 0
        edges = cv2.Canny(grayImg, 100, 255)

        # sharpening_kernel = np.array([[0, -1,  0],
        #                            [-1,  6, -1],
        #                            [0, -1,  0]])

        # grayImg = cv2.filter2D(grayImg, -1, sharpening_kernel)
        # grayImg[grayImg<60] = 0
        # shiftedImg = np.zeros_like(grayImg)
        # shiftedImg[:, 2:] = grayImg[:, :-2]

        # rslt = shiftedImg - grayImg

        # rslt[rslt>0] = 255

        # grayImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
        

        # edges = cv2.Canny(blurred, 0, 255)
        # sobel_y = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=9)

        # grayImg[grayImg>0]  = 255

        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # grayImg  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grayImg, connectivity=8)
        # filtered_image = np.zeros_like(grayImg)

        # for i in range(1, num_labels):  # Start from 1 to skip the background
        #     if stats[i, cv2.CC_STAT_AREA] >= blob_area:
        #         filtered_image[labels == i] = 255        

        # grayImg  = cv2.morphologyEx(grayImg, cv2.MORPH_CLOSE, kernel)

        # edges = cv2.Canny(grayImg, 0, 200)
        # sobel_y = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=9)
        # grayImg = cv2.dilate(filtered_image, kernel, iterations=1)
        
        # contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]
        
        # for i, contour in enumerate(filtered_contours):
        #     print(contour.shape)
        #     print(cv2.contourArea(contour))
        #     contour_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization    
        #     cv2.drawContours(contour_image, filtered_contours, i, (0, 255, 0), 2)
        #     cv2.imshow('', contour_image)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        
        # mask_2 = np.zeros_like(grayImg)
        # mask_2[filtered_image>0] = 255

        # img = cv2.bitwise_and(img,img, mask=mask_2)

        cv2.imshow('', edges)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return img                            

    def processOBB(self, yoloOBB_rslt, conf_thres):
        preds = np.zeros((1,10))
        for result in yoloOBB_rslt:
            boxes  = result.obb.xyxyxyxy.cpu().numpy().reshape(-1,8)
            clsz   = result.obb.cls.cpu().numpy()
            conf   = result.obb.conf.cpu().numpy()
            _preds = np.hstack([clsz[:,None], boxes, conf[:,None]])
            preds  = np.vstack((preds, _preds))
        return preds[preds[:, -1] > conf_thres]

    def predYOLO_obb(self, img, conf=0.5, device='cpu'):  
        results = self.model.predict(img, conf=conf, device=device, verbose=True)
        preds   = self.processOBB(results, conf)
        return preds
    
    def processPins(self, img, preds, surface, adjustment=0):
        '''Post-processes the predicted yolo ROI for pins and separate the upper and lower pins in Front view'''
        pin1 = None
        pin2 = None

        def pinSeparation(_preds, yRefrences):
            '''method to separate the upper and lower pins in front views'''
            lowerPin_thres = np.max(_preds[:,[5,7]])
            # pin1 seperation
            pin1  = preds[np.all(preds[:,[1,3,5,7]]<lowerPin_thres, axis=1)]
            pin1  = pin1[pin1[:,0].argsort()]
            # cap the top y points (top left y, top right y) within a y threshold (190)
            pin1[:,[5,7]] = np.maximum(pin1[:,[5,7]],yRefrences)
            pin1[:,[5,7]] = np.minimum(pin1[:,[5,7]],yRefrences+10)
            # cap the bottom y points (bottom left y, bottom right y) with in a threshold range (710-720)
            # choose a threshold based on the model prediction
            thres = np.max(pin1[:,[1,3]])-5
            pin1[:,[1,3]] = np.maximum(pin1[:,[1,3]],thres)
            pin1[:,[1,3]] = np.minimum(pin1[:,[1,3]],thres)
            # pin2 seperation
            pin2 = preds[np.any(preds[:,[1,3,5,7]]>=lowerPin_thres, axis=1)]
            pin2 = pin2[pin2[:,0].argsort()]
            return pin1, pin2    

        if surface=='Front11':
            pin1, pin2 = pinSeparation(preds, 130)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

            pin2[:,[0,6]] = pin2[:,[0,6]] - adjustment
            pin2[:,[2,4]] = pin2[:,[2,4]] + adjustment
            # # only top y points of pin2 will be capped ()
            # pin2[:,[5,7]] = np.maximum(pin2[:,[5,7]],yRefrences[1]+25)
        elif surface=='Front12':
            pin1, pin2 = pinSeparation(preds, 0)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

            pin2[:,[0,6]] = pin2[:,[0,6]] - adjustment
            pin2[:,[2,4]] = pin2[:,[2,4]] + adjustment

            # self.visPreds(img, pin1, obb=True)
            # self.visPreds(img, pin2, obb=True)
            # # only top y points of pin2 will be capped ()
            # pin2[:,[5,7]] = np.maximum(pin2[:,[5,7]],yRefrences[1]+25)
        elif surface=='Front21':
            # self.visPreds(img, preds, obb=True)
            pin1, pin2 = pinSeparation(preds, 145)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

            pin2[:,[0,6]] = pin2[:,[0,6]] - adjustment
            pin2[:,[2,4]] = pin2[:,[2,4]] + adjustment

            # self.visPreds(img, pin1, obb=True)
            # self.visPreds(img, pin2, obb=True)
            # # only top y points of pin2 will be capped ()
            # pin2[:,[5,7]] = np.maximum(pin2[:,[5,7]],yRefrences[1]+25)
        elif surface=='Front22':
            # self.visPreds(img, preds, obb=True)
            pin1, pin2 = pinSeparation(preds, 0)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment

            pin2[:,[0,6]] = pin2[:,[0,6]] - adjustment
            pin2[:,[2,4]] = pin2[:,[2,4]] + adjustment

            # self.visPreds(img, pin1, obb=True)
            # self.visPreds(img, pin2, obb=True)
            # # only top y points of pin2 will be capped ()
            # pin2[:,[5,7]] = np.maximum(pin2[:,[5,7]],yRefrences[1]+25)    
        elif surface=='Top11' or surface=='Top21':
            # self.visPreds(img, preds, obb=True)
            pin1 = preds

            topThres      = np.average(pin1[:,[5,7]])
            pin1[:,[5,7]] = np.maximum(pin1[:,[5,7]],topThres)
            pin1[:,[5,7]] = np.minimum(pin1[:,[5,7]],topThres)

            bottomThres   = np.max(np.max(pin1[:,[1,3]]))-5
            pin1[:,[1,3]] = np.maximum(pin1[:,[1,3]],bottomThres)
            pin1[:,[1,3]] = np.minimum(pin1[:,[1,3]],bottomThres)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment
            # self.visPreds(img, pin1, obb=True)
        elif surface=='Top12' or surface=='Top22':
            # self.visPreds(img, preds, obb=True)
            pin1 = preds

            topThres      = np.average(pin1[:,[5,7]])
            pin1[:,[5,7]] = np.maximum(pin1[:,[5,7]],topThres)
            pin1[:,[5,7]] = np.minimum(pin1[:,[5,7]],topThres)

            bottomThres   = img.shape[0]-1
            pin1[:,[1,3]] = np.maximum(pin1[:,[1,3]],bottomThres)
            pin1[:,[1,3]] = np.minimum(pin1[:,[1,3]],bottomThres)
            pin1[:,[0,6]] = pin1[:,[0,6]] - adjustment
            pin1[:,[2,4]] = pin1[:,[2,4]] + adjustment
            # self.visPreds(img, pin1, obb=True)

        return pin1, pin2    

    def segment(self,img,obbBoxes, roi='Pin', filterThresh=25):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if roi=='Pin':
            for box in obbBoxes:
                _box = np.array([[int(box[0]), int(box[1])],
                                        [int(box[2]), int(box[3])],
                                        [int(box[4]), int(box[5])],
                                        [int(box[6]), int(box[7])]])    
                _box = _box.reshape(-1,1,2)
                cv2.fillPoly(mask, [_box], color=255)
            roiImg = cv2.bitwise_and(img,img,mask=mask)
            b, g, r = cv2.split(roiImg)
            b[b < filterThresh] = 0
            g[g < filterThresh] = 0
            r[r < filterThresh] = 0
            roiImg  = cv2.merge((b, g, r))
            roiMask = mask
            roiBox  = ''
        elif roi=='Bur':
            # _img = copy.deepcopy(img)
            # # self.visPreds(_img, obbBoxes, obb=True)
            # for box in obbBoxes:
            #     x1 = int(box[6]) -10 if (int(box[6]) -10)>0 else 0
            #     x2 = int(box[2]) +10
            #     y1 = int(box[7]) -10 if (int(box[7]) -10)>0 else 0
            #     y2 = int(box[3]) +10 if (int(box[3]+10) < img.shape[0]) else img.shape[0]
            #     pinImg = img[y1:y2, x1:x2]        # TopL (6,7), BotomR (2,3)

            #     grayPin_img = cv2.cvtColor(pinImg, cv2.COLOR_BGR2GRAY)
            #     grayPin_img[grayPin_img>50] = 255

            #     y = int(((y2-y1)/3)*2)
            #     _y = (y2-y1) - 20
            #     row = grayPin_img[y, :]
            #     x = np.argmax(row==255)-2
            #     _x = len(row) - 1 - np.argmax(row[::-1] == 255)

            #     cv2.line(grayPin_img, (x,y), (_x,y), (0,0,255))
            #     cv2.line(grayPin_img, (x,_y), (_x,_y), (0,0,255))

            #     # grayPin_img = 255 - grayPin_img

            #     # edges = cv2.Canny(grayPin_img, 0, 255)

            #     # kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            #     # edges = cv2.morphologyEx(grayPin_img, cv2.MORPH_ERODE, kernel)

            #     cv2.imshow('', grayPin_img)
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()
            roiBox = self.ROIbox(obbBoxes, padding=(30,5))
            # mask[roiBox[1]:roiBox[3],roiBox[0]:roiBox[2]] = 255
            # roiImg = cv2.bitwise_and(img,img,mask=mask)
            roiImg  = np.zeros_like(img, dtype=np.uint8)
            roiMask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            roiImg[roiBox[1]:roiBox[3],roiBox[0]:roiBox[2]] = img[roiBox[1]:roiBox[3],roiBox[0]:roiBox[2]]
            roiMask[roiBox[1]:roiBox[3],roiBox[0]:roiBox[2]]= 255
            # roiImg[np.where(mask==0)] = 0
            # mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for box in obbBoxes:
                _box = np.array([[int(box[0]), int(box[1])],
                                        [int(box[2]), int(box[3])],
                                        [int(box[4]), int(box[5])],
                                        [int(box[6]), int(box[7])]])    
                _box = _box.reshape(-1,1,2)
                cv2.fillPoly(mask, [_box], color=255)
            # roiImg = cv2.bitwise_and(roiImg,roiImg,mask=~mask)
            roiImg[np.where(mask>0)]  = (150,230,240)
            roiMask[np.where(mask>0)] = 0       
        return roiImg, roiMask, roiBox    

    def ROIbox(self, obbPreds, width=1920, height=1280, padding=(20,20)):
        x_min = int(np.min(obbPreds[:, [0, 2, 4, 6]])) - padding[0]
        y_min = int(np.min(obbPreds[:, [1, 3, 5, 7]])) - padding[1]
        x_max = int(np.max(obbPreds[:, [0, 2, 4, 6]])) + padding[0]
        y_max = int(np.max(obbPreds[:, [1, 3, 5, 7]])) + padding[1]

        x_min = np.clip(x_min, 0, width - 1)
        y_min = np.clip(y_min, 0, height - 1)
        x_max = np.clip(x_max, 0, width - 1)
        y_max = np.clip(y_max, 0, height - 1)

        return [x_min, y_min, x_max, y_max]
    
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
    
    def pinROI(self, img, surface):
        # Pin ROI detection
        # pp_img = self.preProcess(img, surface)
        preds = self.predYOLO_obb(img)        
        preds = preds[:,1:9]
        
        # # ROI post processing
        # if surface=='Front1':
        #     pin1, pin2 = self.processPins(preds, 'Front1', (190,725))
        # elif surface=='Front2':
        #     pin1, pin2 = self.processPins(preds, 'Front2', ((190,745)))
        #     # self.visPreds(_img1, preds, obb=True)
        # elif surface=='Top1' or surface=='Top2':
        #     pin1, pin2 = self.processPins(preds, 'Top1', (225,635), 1)

        if surface=='Front11':
            roi     = [75, 95, 1835, 875]      # to filter out unwanted predictions not in the ROI
            preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Front11', -1)
        elif surface=='Front12':
            # roi     = [120, 0, 1835, 690]
            # preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Front12', -1)    
        elif surface=='Front21':
            roi     = [115, 95, 1850, 875]
            preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Front21', -1)
        elif surface=='Front22':
            # roi     = [140, 0, 1840, 680]
            # preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Front22', -1)    
            # self.visPreds(_img1, preds, obb=True)
        elif surface=='Top11':
            roi     = [120, 520, 1895, 1225]
            # self.visPreds(img, preds, obb=True)
            preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Top11', -3)
        elif surface=='Top12':
            # roi     = [120, 710, 1890, 1280]
            # preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Top12', -3)
        elif surface=='Top21':
            roi = [125, 520, 1905, 1215]
            # self.visPreds(img, preds, obb=True)
            preds   = self.filterPreds(preds, roi)
            # self.visPreds(img, preds, obb=True)
            pin1, pin2 = self.processPins(img, preds, 'Top21', -3)
        elif surface=='Top22':
            # roi     = [150, 690, 1915, 1280]
            # cv2.rectangle(img, (roi[0],roi[1]), (roi[2],roi[3]), (0,0,255), 2)
            # cv2.imshow('', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # preds   = self.filterPreds(preds, roi)
            pin1, pin2 = self.processPins(img, preds, 'Top22', -3)   

        # visualize pin ROIs
        # self.visPreds(_img1, pin1, obb=True)
        # self.visPreds(_img2, pin2, obb=True)

        # ROI segmentation (based on the refined pin ROIs)
        # Here only the pin region will be retained in the image
        if 'Front' in surface:
            upperPin_box = self.ROIbox(pin1)
            
            lowerPin_box = self.ROIbox(pin2)
             
            fullPin_img, fullMask, _ = self.segment(img, np.vstack((pin1,pin2)), 'Pin')
            Pin_img     = fullPin_img[upperPin_box[1]:upperPin_box[3], upperPin_box[0]:upperPin_box[2]]
            Pin_img2    = fullPin_img[lowerPin_box[1]:lowerPin_box[3], lowerPin_box[0]:lowerPin_box[2]]
            pinBox      = [upperPin_box, lowerPin_box]
        elif 'Top' in surface:
            pinBox = self.ROIbox(pin1)
            fullPin_img, fullMask, _ = self.segment(img, pin1, 'Pin', 30)
            Pin_img     = fullPin_img[pinBox[1]:pinBox[3], pinBox[0]:pinBox[2]]
            Pin_img2    = None
        return fullPin_img, Pin_img, Pin_img2, pinBox, pin1, pin2, fullMask   
    
    def burrROI(self, img, surface):
        # Pin ROI detection
        # pp_img = self.preProcess(img, surface)
        preds = self.predYOLO_obb(img)        
        preds = preds[:,1:9]
        # ROI post processing
        # for old images-----------------------------------------------
        # if surface=='Front1':
        #     roi     = [40, 140, 1860, 960]
        #     preds   = self.filterPreds(preds, roi)
        #     pin1, _ = self.processPins(preds, 'Front1', (190,725), -2)
        # elif surface=='Front2':
        #     roi     = [40, 140, 1860, 960]
        #     preds   = self.filterPreds(preds, roi)
        #     pin1, _ = self.processPins(preds, 'Front2', (190,745), -2)
        #     # self.visPreds(_img1, preds, obb=True)
        # elif surface=='Top1' or surface=='Top2':
        #     roi     = [100, 550, 1920, 1280]
        #     preds   = self.filterPreds(preds, roi)
        #     pin1, _ = self.processPins(preds, 'Top1', (225,635), -2)

        # for new images--------------------------------------------------
        if surface=='Front11':
            roi     = [75, 95, 1835, 875]      # to filter out unwanted predictions not in the ROI
            preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Front11', -1)
        elif surface=='Front12':
            # roi     = [120, 0, 1835, 690]
            # preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Front12', -1)    
        elif surface=='Front21':
            roi     = [115, 95, 1850, 875]
            preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Front21', -1)
        elif surface=='Front22':
            # roi     = [140, 0, 1840, 680]
            # preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Front22', -1)    
            # self.visPreds(_img1, preds, obb=True)
        elif surface=='Top11':
            roi     = [120, 520, 1895, 1225]
            preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Top11', -2)
        elif surface=='Top12':
            # roi     = [120, 710, 1890, 1280]
            # preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Top12', -2)
        elif surface=='Top21':
            roi = [125, 520, 1905, 1215]
            preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Top21', -2)
        elif surface=='Top22':
            # roi     = [150, 690, 1915, 1280]
            # cv2.rectangle(img, (roi[0],roi[1]), (roi[2],roi[3]), (0,0,255), 2)
            # cv2.imshow('', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # preds   = self.filterPreds(preds, roi)
            pin1, _ = self.processPins(img, preds, 'Top22', -2)            
              

        # visualize pin ROIs
        # self.visPreds(img, pin1, obb=True)
        # self.visPreds(img, pin2, obb=True)

        # ROI segmentation (based on the refined pin ROIs)
        # Here only the Bur region will be retained in the image
        # ROIbox  = self.ROIbox(pin1, padding=(30,30))
        fullImg, fullMask, ROIbox = self.segment(img, pin1, 'Bur')
        burImg  = fullImg[ROIbox[1]:ROIbox[3], ROIbox[0]:ROIbox[2]]
        burMask = fullMask[ROIbox[1]:ROIbox[3], ROIbox[0]:ROIbox[2]]
        # ROIbox  = [f'{int(ROIbox[0])} {int(ROIbox[1])} {int(ROIbox[2])} {int(ROIbox[3])}']
        return fullImg, fullMask, burImg, burMask, ROIbox, pin1

if __name__=='__main__':        
    # _path      = '/home/zafar/old_pc/data_sets/robot-project-datasets/TML_bur_original/2024-12-23_original/un_confirmed/front'
    _path      = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/Abrasion/Top-pin_auto_1'
    # _path      = '/home/zafar/old_pc/data_sets/robot-project-datasets/roi-test-exp'
    # _path      = '/home/zafar/old_pc/data_sets/robot-project-datasets/SystemStart-20250110-151428/selected_data/all_anomaly'
    # _path      = '/home/zafar/old_pc/data_sets/robot-project-datasets/SystemStart-20250110-151428/selected_data/Top-pin_22'
    model_path = 'runs/TML-Pin-ROI/coco-pretrain/Exp1/train7/weights/best.pt'
    savePath   = '/home/zafar/old_pc/data_sets/mvtec_anomaly_detection/Abrasion-Scratch/test/'
    roiModel   = ROI_model(model_path)

    pathList = glob.glob(f'{_path}/*.png')
    idx = 0
    
    for imgPath in pathList:
        name    = imgPath.split('/')[-1].split('Input-')[-1].split('__Cam')[0]
        surface = None

        # for old images------------------
        # if 'Front1' in imgPath:
        #     surface = 'Front1'
        # elif 'Front2' in imgPath:
        #     surface = 'Front2'
        # elif 'Top1' in imgPath:
        #     surface = 'Top1'
        # elif 'Top2' in imgPath:
        #     surface = 'Top2'

        # for new images---------------------------------------------------------
        
        # if 'Front-pin_11' in imgPath:       # front surface 1, focus on first row
        #     surface = 'Front11'
        # elif 'Front-pin_12' in imgPath:     # front surface 1, focus on second row
        #     surface = 'Front12'    
        # elif 'Front-pin_21' in imgPath:     # front surface 2, focus on first row
        #     surface = 'Front21'
        # elif 'Front-pin_22' in imgPath:     # front surface 2, focus on second row
        #     surface = 'Front22'    
        # elif 'Top-pin_11' in imgPath:       # top surface 1, focus on first row
        #     surface = 'Top11'
        # elif 'Top-pin_12' in imgPath:       # top surface 1, focus on second row
        #     surface = 'Top12'    
        # elif 'Top-pin_21' in imgPath:       # top surface 2, focus on first row     
        #     surface = 'Top21'
        # elif 'Top-pin_22' in imgPath:       # top surface 2, focus on second row
        #     surface = 'Top22'

        
        if "Front-pin-2nd_auto_0" in imgPath:
            surface = 'Front12'
        elif "Front-pin-2nd_auto_1" in imgPath:
            surface = 'Front22'
        elif "Front-pin_auto_0" in imgPath:
            surface = 'Front21'
        elif "Front-pin_auto_1" in imgPath:
            surface = 'Front11'
        elif "Top-pin-2nd_auto_0" in imgPath:
            surface = 'Top12'
        elif "Top-pin-2nd_auto_1" in imgPath:
            surface = 'Top22'
        elif "Top-pin_auto_0" in imgPath:
            surface = 'Top21'
        elif "Top-pin_auto_1" in imgPath:
            surface = 'Top11'                

        if surface:
            img  = cv2.imread(imgPath)
            img  = cv2.resize(img, (1920,1280))
            # cv2.line(img, (0,460), (1920,460), (0,255,0), 2)      # 22
            # cv2.line(img, (190,0), (190,1280), (0,255,0), 2)
            # cv2.line(img, (0,644), (1920,644), (0,255,0), 2)      # 21
            # cv2.line(img, (175,0), (175,1280), (0,255,0), 2)
            # cv2.line(img, (0,652), (1920,652), (0,255,0), 2)        # 11
            # cv2.line(img, (152,0), (152,1280), (0,255,0), 2)
            # cv2.line(img, (0,476), (1920,476), (0,255,0), 2)        # 12
            # cv2.line(img, (164,0), (164,1280), (0,255,0), 2)
            # roiModel.visPreds(_img,pinRoi1,boxType='xyxy')
            # cv2.imshow('', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # TODO: get back the box region for accurateley draw predictions on to real image
            fullPin_img, Pin_img, Pin_img2, pinBox, pin1, pin2, fullMask = roiModel.pinROI(img, surface)

            # cv2.imshow('', img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            sorted_indices = np.argsort(pin1[:, 6])
            sortedPins = pin1[sorted_indices]
            # roiModel.visPreds(fullPin_img, sortedPins, obb=True)

            if surface=='Top21':
                if not os.path.exists(f'{savePath}train/good'):
                    os.makedirs(f'{savePath}train/good')
                if not os.path.exists(f'{savePath}fg_mask'):
                    os.makedirs(f'{savePath}fg_mask')

                crop1 = sortedPins[0:4, :]
                top = crop1[0,-2:]
                bottom = crop1[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                
                crop2 = sortedPins[4:8, :]
                top = crop2[0,-2:]
                bottom = crop2[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                crop3 = sortedPins[8:12, :]
                top = crop3[0,-2:]
                bottom = crop3[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                
                crop4 = sortedPins[12:16, :]
                top = crop4[0,-2:]
                bottom = crop4[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                crop5 = sortedPins[16:, :]
                top = crop5[0,-2:]
                bottom = crop5[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                # print()
            elif surface=='Top11':
                if not os.path.exists(f'{savePath}train/good'):
                    os.makedirs(f'{savePath}train/good')
                if not os.path.exists(f'{savePath}fg_mask'):
                    os.makedirs(f'{savePath}fg_mask')

                crop1 = sortedPins[0:6, :]
                top = crop1[0,-2:]
                bottom = crop1[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                
                crop2 = sortedPins[6:12, :]
                top = crop2[0,-2:]
                bottom = crop2[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                crop3 = sortedPins[12:18, :]
                top = crop3[0,-2:]
                bottom = crop3[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                
                crop4 = sortedPins[18:21, :]
                top = crop4[0,-2:]
                bottom = crop4[-1,2:4]
                crop = fullPin_img[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                crop = cv2.resize(crop, (1024,1024))
                mask = fullMask[int(top[1]-10):int(bottom[1]+10), int(top[0]-10): int(bottom[0]+10)]
                mask = cv2.resize(mask, (1024,1024))
                imgName = savePath+'train/good/'+name+f'_{idx}.png'
                MaskName = savePath+'fg_mask/'+name+f'_{idx}.png'
                cv2.imwrite(imgName, crop)
                cv2.imwrite(MaskName, mask)
                idx+=1
                # cv2.imshow('', crop)
                # cv2.waitKey()
                # cv2.destroyAllWindows()


                print()
            # cv2.imshow('', fullPin_img)
            # cv2.waitKey()
            # cv2.imshow('',Pin_img)
            # cv2.waitKey()
            # if Pin_img2 is not None:
            #     cv2.imshow('', Pin_img2)
            #     cv2.waitKey()
            # cv2.destroyAllWindows()

            # print()

            # fullImg, fullMask, burImg, burMask, ROIbox = roiModel.burrROI(img, surface)
            # roiModel.burrROI(img, surface)

            # savenameFull_img = f'{savePath}/fullImg/{name}'
            # savenameFull_mask = f'{savePath}/fullmask/{name}' 
            # savenameBurr_img = f'{savePath}/burrROI_img/{name}'
            # savenameBurr_mask = f'{savePath}/burrROI_mask/{name}'
            # savenameBox  = f'{savePath}/fullImg/{name.split(".")[0]}.txt'    
            
            # burImg = cv2.resize(burImg, (1024,1024))
            # savenameBurr_img = f'{savePath}/burr-full/{name}'
            # savenameBurr_mask = f'{savePath}/fg_mask_full/{name}'
            # cv2.imwrite(savenameBurr_img, burImg)
            # cv2.imwrite(savenameBurr_mask, burMask)
            # if not os.path.exists(f'{savePath}/fullImg'):
            #     os.makedirs(f'{savePath}/fullImg')
            # if not os.path.exists(f'{savePath}/fullmask'):
            #     os.makedirs(f'{savePath}/fullmask')        
            # xSplit = round(burImg.shape[1]/3)
            # x1     = 0
            # for i in range(3):
            #     x2 = xSplit if xSplit<burImg.shape[1] else burImg.shape[1]
            #     img = burImg[0:burImg.shape[0], x1:xSplit]
            #     mask = burMask[0:burImg.shape[0], x1:x2]
            #     img = cv2.resize(img, (1024,1024))
            #     mask = cv2.resize(mask, (1024,1024))
            #     x1 = xSplit
            #     xSplit += xSplit
            #     savenameBurr_img = f'{savePath}/burr/{i}_{name}'
            #     savenameBurr_mask = f'{savePath}/fg_mask/{i}_{name}'
            #     cv2.imwrite(savenameBurr_img, img)
            #     cv2.imwrite(savenameBurr_mask, mask)
                # cv2.imshow('', img)
                # cv2.waitKey()
                # cv2.imshow('', mask)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
            # cv2.imwrite(savenameFull, fullImg)
            # cv2.imwrite(savenameBurr, burImg)

            # with open(savenameBox, 'w') as f:
            #     f.writelines(roiBox)

            # cv2.imshow('', fullImg)
            # cv2.waitKey()
            # cv2.imshow('', fullMask)
            # cv2.waitKey()
            # cv2.imshow('', burImg)
            # cv2.waitKey()
            # cv2.imshow('', burMask)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

# kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# grayImg = cv2.morphologyEx(grayImg, cv2.MORPH_CLOSE, kernel)
# grayImg = cv2.dilate(grayImg, kernel, iterations=1)
# grayImg = cv2.erode(grayImg, kernel, iterations=1)

# blurred = cv2.GaussianBlur(grayImg, (5, 5), 0)
# grayImg = cv2.dilate(filtered_image, kernel, iterations=1)

# edges = cv2.Canny(filtered_image, 0, 255)
# sobel_y = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=9)

