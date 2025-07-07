import cv2
import numpy as np
import torch
import threading

# TODO 1: Import your model here. Using YOLO as example
from ultralytics import YOLO

# TODO 2:Import any other packages required by your code
import time
from loguru import logger
import datetime
import emoji
from colorama import Fore, Style
# from svgwrite import Drawing
# from svgwrite.container import Group
# from svgwrite.shapes import Rect
# from svgwrite.text import Text

import sys
import os
import json

# from utils.colors import red

# from utils.colors import white

# src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
# sys.path.append(src_dir)
from abr_ROI import MODEL as roiModel

class MODEL:
    """
    A class that encapsulates the model loading, warm-up, and inference operations.
    This approach ensures the model is loaded into memory once and used for multiple inferences.
    """
    def __init__(self, 
                #  brightnessConfig    = '/home/gpuadmin/Desktop/WEIGHTS/08_Abrasion/brightness_config.json',
                 model_path            = '', 
                 confidence_threshold  = 0.3,
                 show_blue_boxes       = True, 
                 annotate_SVG          = True,
                 debug                 = False, 
                 warmup_runs           = 1, 
                 roiModel_path         = None,
                 roiModel_confidence   = None,
                 device_id             = 0,
                 itrs                  = 2,
                 erosion_kernel        = 5
                 ):  #TODO 3: Update the model_path, Add the parameters you need which can be modified from GUI
        """
        Initialize the model.

        Args:
            model_path (str): Path to the YOLO model file.
            confidence_threshold (float): Confidence threshold for detections.
            warmup_runs (int): Number of dummy inference runs to warm up the model.
        """

        try:
            num_gpus    = torch.cuda.device_count()
            self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and num_gpus > device_id else "cuda:0")
            # self.device_id  = device_id
            
            self.roiModel_path       = roiModel_path
            self.roiModel_confidence = roiModel_confidence
            self.debug               = debug
            self.itrs                = itrs
            self.erosion_kernel      = erosion_kernel
            
            self.logger = logger.bind(model="m08_Abrasion")
            try:
                path = os.path.join("logs/models/", "m08_Abrasion", "m08_Abrasion.log")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.logger.add(path, rotation="1 day", level="TRACE",
                                retention="60 days", compression="zip",
                                enqueue=True, backtrace=True, diagnose=True, colorize=False)
                self.logger.info(f"Logging to {path}")
            except:
                self.logger.add(f"src/ai_vision/logs/m08_Abrasion{datetime.datetime.now()}.log", rotation="10 MB", level="INFO")
            
            self.roiModel = roiModel(model_path=self.roiModel_path,
                                    confidence_threshold=self.roiModel_confidence,
                                    device_id=device_id)
            
            # Detection parameters
            self.kernel9      = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
            self.kernel7      = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
            self.kernel5      = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            self.kernel3      = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            self.kernel59     = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
            self.kernel95     = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
            # self.kernel4            = cv2.getStructuringElement(cv2.MORPH_RECT,(2,5))
            self.showFiltered = show_blue_boxes
            self.annotate_SVG = annotate_SVG
            self.anom_width   = 0.1
            self.anom_height  = 0.5
            self.denom        = 80
            self.conf_thres   = confidence_threshold
            self.model_lock   = threading.Lock()

            self.model = YOLO(model_path).to(self.device)          
            
            # TODO 6: Optional warm-up for improved inference speeds on the first real run
            # Optional warm-up for improved inference speeds on the first real run
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(warmup_runs):
                _ = self.model.predict(source=dummy_image, conf=self.conf_thres, verbose=False, device=self.device)
            #     _ = self.model_2.predict(source=dummy_image, conf=self.conf_thres, verbose=False, device=self.device)
        except Exception as e:
            self.logger.exception(f"Error initializing model: {e}")
            raise e

    def drawBox(self, img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1, adj=0, label=False):
        if obb:
            # print('visualizing obb preds')
            for box in boxes:
                _box = np.array([[int(box[0])-adj, int(box[1])-adj],
                                [int(box[2])+adj, int(box[3])-adj],
                                [int(box[4])+adj, int(box[5])+adj],
                                [int(box[6])-adj, int(box[7])+adj]])
                _box = _box.reshape(-1,1,2)
                cv2.polylines(img, [_box], isClosed=True, color=color, thickness=thickness)
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

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    # text = f'{box[0]}:'
                    # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                    # label_bg_top_left = (x1, y1 - h - 5)
                    # label_bg_bottom_right = (x1 + w, y1)
                    # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    # cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)

            elif boxType=='xyxy':   # TODO: implement the box rendering for xyxy box coordinates
                for i, box in enumerate(boxes):
                    x1 = int(box[0])
                    x2 = int(box[2])
                    y1 = int(box[1])
                    y2 = int(box[3])

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    if label:
                        cv2.putText(img, f'{i+1}', (x1+2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=color, thickness=thickness)

                    # text = f'{box[0]}:'
                    # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, thickness+1)
                    # label_bg_top_left = (x1, y1 - h - 5)
                    # label_bg_bottom_right = (x1 + w, y1)
                    # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    # cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness+1)

    def _putText(self, img, text='', pos=(15,15), thick=5, fontSize=3, color=(0,0,255), bold=True):
        text = f'{text}'
        # (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 5, thick+1)
        # label_bg_top_left = (pos[0], pos[1] - h - 5)
        # label_bg_bottom_right = (pos[0] + w, pos[1])
        # cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
        if bold:
            cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_DUPLEX, fontSize, color, thick+1)
        else:
            cv2.putText(img, text, (pos[0], pos[1]), cv2.FONT_HERSHEY_SIMPLEX, fontSize, color, thick)    
    
    def getPin(self, img, pinOBB, kernel):
        pinTop    = [int(pinOBB[6])-2, int(pinOBB[7])]
        pinBottom = [int(pinOBB[2])+2, int(pinOBB[3])]
        pinImg    = img[pinTop[1]:pinBottom[1], pinTop[0]:pinBottom[0]]
        h,_,_     = pinImg.shape
        adjustedH = int(h*0.95)
        pinMask   = cv2.cvtColor(pinImg, cv2.COLOR_BGR2GRAY)

        pinMask[pinMask>0] = 255
        pinMask = cv2.morphologyEx(pinMask, cv2.MORPH_CLOSE, kernel)
        pinMask[adjustedH:,:] = 0
        pinImg = cv2.bitwise_and(pinImg,pinImg,mask=pinMask)
        # pinImg = pinImg[pinImg>0]
        return pinImg

    def brightness_level(self,img):
        sum1 = np.sum(img[:,:,0])
        sum2 = np.sum(img[:,:,1])
        sum3 = np.sum(img[:,:,2])
        cum_sum = sum1 + sum2 + sum3
        # Formula to calculate the brightness of an image:
        # 2 * (cumulative sum of all pixel intensities(in every channels) / total pixels in image x 3x 255)
        brightness = 2*(cum_sum/(img.shape[0]*img.shape[1]*3*255))         
        return brightness   
    
    def filterRed(self, hsv_image):
        # h, w, _ = patch.shape
        # # Convert to HSV color space
        # hsv_image = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        # Red hue wraps around in HSV (0–1 scale): [0.94–1.0] U [0.0–0.06]
        hue = hsv_image[:, :, 0]/180
        sat = hsv_image[:, :, 1]/255
        val = hsv_image[:, :, 2]/255

        # Red hue masks
        mask1 = (hue > 0.96) & (hue <= 1.00) # & (sat > 0.4) & (val > 0.3)
        mask2 = (hue >= 0.0) & (hue < 0.09) #& (sat > 0.2) & (val > 0.4)

        # Saturation filter to exclude low-color areas
        saturation_mask = sat > 0.2
        value_mask      = (val > 0.2) # & (val < 0.98)

        # Final red mask (logical OR on hue, AND with saturation)
        mask_red = (mask1 | mask2) & saturation_mask & value_mask

        # filteredRed = cv2.bitwise_and(patch, patch, mask=mask_red.astype(np.uint8) * 255)
        return mask_red

    def filterYellow(self, hsv_image):
        # h, w, _ = patch.shape
        # Convert to HSV color space
        # hsv_image = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        hue = hsv_image[:, :, 0]/180
        sat = hsv_image[:, :, 1]/255
        val = hsv_image[:, :, 2]/255

        # Yellow typically ranges from about 0.09 to 0.25 in HSV (OpenCV range scaled 0–1)
        mask_yellow = (hue > 0.07) & (hue < 0.17)
        saturation_mask = sat > 0.2
        value_mask      = (val > 0.2) # & (val < 0.98)

        # Final yellow mask
        mask_yellow = mask_yellow & saturation_mask & value_mask
        # filteredYellow = cv2.bitwise_and(patch, patch, mask=mask_yellow.astype(np.uint8) * 255)
        return mask_yellow
    
    def filterOrange(self, hsv_image):
        # h, w, _ = patch.shape
        # Convert to HSV color space
        # hsv_image = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        hue = hsv_image[:, :, 0]/180
        sat = hsv_image[:, :, 1]/255
        val = hsv_image[:, :, 2]/255

        # Orange typically ranges from about 0.01 to 0.09 in HSV (OpenCV range scaled 0–1)
        mask_orange = (hue > 0.01) & (hue < 0.12)
        saturation_mask = sat > 0.1
        value_mask      = (val > 0.2) # & (val < 0.98)

        # Final orange mask
        mask_orange = mask_orange & saturation_mask & value_mask
        # filteredOrange = cv2.bitwise_and(patch, patch, mask=mask_orange.astype(np.uint8) * 255)
        return mask_orange
    
    def filterBrown(self, hsv_image):
        # h, w, _ = patch.shape
        # Convert to HSV color space
        # hsv_image = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        hue = hsv_image[:, :, 0]/180
        sat = hsv_image[:, :, 1]/255
        val = hsv_image[:, :, 2]/255

        # Brown typically ranges from about 0.04 to 0.16 in HSV (OpenCV range scaled 0–1)
        mask_brown      = (hue > 0.07) & (hue < 0.175)
        saturation_mask = (sat > 0.05) & (sat < 0.7)
        value_mask      = (val > 0.1) & (val < 0.75)

        # Final brown mask
        mask_brown = mask_brown & saturation_mask & value_mask
        # filteredBrown = cv2.bitwise_and(patch, patch, mask=mask_brown.astype(np.uint8) * 255)
        return mask_brown
    
    def filterCopper(self, hsv_image):
        # h, w, _ = patch.shape
        # Convert to HSV color space
        # hsv_image = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        hue = hsv_image[:, :, 0]/180
        sat = hsv_image[:, :, 1]/255
        val = hsv_image[:, :, 2]/255

        # Copper typically ranges from about 0.05 to 0.09 in HSV (OpenCV range scaled 0–1)
        mask_copper = (hue > 0.03) & (hue < 0.09)
        saturation_mask = (sat > 0.05) # & (sat < 0.98)
        value_mask      = (val > 0.2) # & (val < 0.98)

        # Final copper mask
        mask_copper = mask_copper & saturation_mask & value_mask
        # filteredCopper = cv2.bitwise_and(patch, patch, mask=mask_copper.astype(np.uint8) * 255)
        return mask_copper
    
    def filterWhite(self, hsv_image):
        # h, w, _ = patch.shape
        # Convert to HSV color space
        # hsv_image = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

        # hue = hsv_image[:, :, 0]/180
        # sat = hsv_image[:, :, 1]/255
        val = hsv_image[:, :, 2]/255

        # val = (np.clip(val * 1.1, 0, 255).astype(np.uint8))

        # White typically has low saturation and high value
        # saturation_mask = (sat < 0.6) # & (sat > 0.1)
        # value_mask      = val > 0.7

        mask_white = (val > 0.75) # & (sat < 0.125)

        # Final white mask
        # mask_white = saturation_mask & value_emask
        # filteredWhite = cv2.bitwise_and(patch, patch, mask=mask_white.astype(np.uint8) * 255)
        return mask_white

    def xyxyToxywh(self, dets):
        _dets = {}
        for i,box in enumerate(dets):
            xc = ((box[0]+box[2])/2).astype(float)
            yc = ((box[1]+box[3])/2).astype(float)
            w  = (abs(box[2]-box[0])).astype(float)
            h  = (abs(box[3]-box[1])).astype(float)
            conf = box[-4]
            _dets[i] = {'bbox': [xc, yc, w, h], 'confidence': conf}
        return _dets
    
    def annotateSVG(
                    self,
                    dims               = (1280,1920),
                    preds_selected     = [],
                    preds_discarded    = [],
                    ignored_pins       = [],
                    showFiltered       = True,
                    # smallerROI         = [],
                    largerROI          = [],
                    y_limit            = 1280,
                    pos1               = (0,0),
                    pos2               = (0,0),
                    pos3               = (0,0),
                    warning           = False,
                    warningText        = '',

                    ):
        
        Anomaly = False
        drawing = SvgDraw(width=dims[1], height=dims[0])
        if not warning:
            # Draw the normal ROI in yellow
            drawing.rectangle(pt1=tuple(map(int, largerROI[:2])),
                            pt2=tuple(map(int, largerROI[2:])),
                            color=(0,0,255),
                            thickness=6)
            
            # Add annotations and label the drwaing
            drawing.putText(text="Dimensions (mm): ", org=pos3, color=(0,0,255), fontScale=3, scale=24, bold=True)
            drawing.putText(text=f"({self.anom_height}x{self.anom_width})mm", org=(pos3[0]+650, pos3[1]), color=(0,255,255), fontScale=3, scale=20)
            drawing.rectangle(pt1=(pos3[0]+1020,pos3[1]-40),pt2=(pos3[0]+1060, pos3[1]), color=(0,255,255), thickness=7)
            
            if len(ignored_pins)>0:
                for pin in ignored_pins:
                    x1, y1, x2, y2 = tuple(map(int, pin[:4]))
                    drawing.rectangle(pt1=(x1, y1), pt2=(x2, y2), color=(255,0,255), thickness=6)
            
            if len(preds_selected)>0:

                Anomaly = True

                # preds_selected[:,:2] -= 2
                # preds_selected[:,2:] += 2

                drawing.putText(text="NG: ", org=pos1, color=(0,0,255), fontScale=3, scale=34, bold=True)
                drawing.putText(text=f"(TML Abrasion & Scratch) ({len(preds_selected)})", org=pos2, color=(0,0,255), fontScale=3, scale=30, bold=True)
                
                _pos   = [pos3[0], pos3[1]+90]
                offset = 650
                idx1   = _pos[1]
                idx2   = 0

                for i, pred in enumerate(preds_selected):
                    x1, y1, x2, y2 = tuple(map(int, pred[:4]))
                    drawing.rectangle(pt1=(x1, y1), pt2=(x2, y2), color=(0,0,255), thickness=5)
                    drawing.putText(text=f'{i+1}', org=(int((x1+x2)/2), int((y1-4))), color=(0,0,255), fontScale=2, scale=22)
                    # dim = areaDims_selected[i]
                    temp1 = pred[-3]
                    temp2 = pred[-2]
                    temp3 = pred[-1]
                    temp4 = pred[-4]
                    drawing.putText(text=f'{i+1}: {temp1:.2f}x{temp2:.2f}, ({temp3:.1f}), ({temp4:.2f})', org=(_pos[0], _pos[1]+(65*idx2)), color=(0,0,255), fontScale=2, scale=26)
                    idx1 += 65*idx2
                    idx2 += 1
                    if idx1>y_limit-20:
                        _pos  = [pos3[0]+offset, pos3[1]+90]
                        idx1  = _pos[1]
                        idx2  = 0
                        offset += offset
            else:
                drawing.putText(text="Good:", org=pos1, color=(0,255,0), fontScale=3, scale=34, bold=True)
            
            if showFiltered and len(preds_discarded)>0:
                # preds_discarded[:,:2] -= 2
                # preds_discarded[:,2:] += 2
                drawing.putText(text=f"({len(preds_discarded)})", org=(pos2[0]+1500, pos2[1]), color=(255,0,0), fontScale=3, scale=30)
                
                _pos   = [pos3[0]+1900, pos3[1]+90]
                offset = 650
                idx1   = _pos[1]
                idx2   = 0

                for i, pred in enumerate(preds_discarded):
                    x1, y1, x2, y2 = tuple(map(int, pred[:4]))
                    drawing.rectangle(pt1=tuple(map(int,[x1,y1])), pt2=tuple(map(int,[x2,y2])), color=(255,0,0), thickness=5)
                    drawing.putText(text=f'{i+1}', org=tuple(map(int,[(x1+x2)/2, y1-4])), color=(255,0,0), fontScale=2, scale=22)
                    # dim = areaDims_discarded[i]
                    temp1 = pred[-3]
                    temp2 = pred[-2]
                    temp3 = pred[-1]
                    temp4 = pred[-4]
                    drawing.putText(text=f'{i+1}: {temp1:.2f}x{temp2:.2f}, ({temp3:.1f}), ({temp4:.2f})', org=(_pos[0], _pos[1]+(65*idx2)), color=(255,0,0), fontScale=2, scale=26)
                    idx1 += 65*idx2
                    idx2 += 1
                    if idx1>y_limit-20:
                        _pos  = [pos3[0]+1900+offset, pos3[1]+80]
                        idx1  = _pos[1]
                        idx2  = 0
                        offset += offset
            # drawing.save('1.svg')
        elif warning:
            drawing.putText(text="Warning: ", org=pos1, color=(0,0,255), fontScale=3, scale=40, bold=True)
            drawing.putText(text=f"{warningText}", org=pos2, color=(0,0,255), fontScale=3, scale=36)
            drawing.putText(text="(TML Abrasion & Scratch)", org=pos3, color=(0,0,255), fontScale=2, scale=28)    
        return drawing.tostring(), Anomaly           

    def visualize(self, patch, patch_pp, whiteMask, colorMask, l, w, colorRatio, save_path="color_segmentation.png"):
        import matplotlib.pyplot as plt
        """
        Saves a visualization showing the original patch, white mask and color mask side by side.

        Args:
            patch: The original image patch (BGR)
            whiteMask: Binary mask for white pixels
            colorMask: Binary mask for colored pixels
            titles: List of color names
            save_path: Output image file path (e.g., 'output.png')
        """
        fig, axs = plt.subplots(1, 4, figsize=(15, 8))
        fig.suptitle("Abrasion Detection Visualization", fontsize=10)

        # Plot original patch
        axs[0].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Prediction Patch", fontsize=8)
        axs[0].axis('off')

        # Plot processed patch
        axs[1].imshow(cv2.cvtColor(patch_pp, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Processed Patch", fontsize=8)
        axs[1].axis('off')

        # Plot white mask
        axs[2].imshow(whiteMask, cmap='gray')
        axs[2].set_title("White Mask", fontsize=8)
        axs[2].axis('off')

        # Plot color mask
        axs[3].imshow(colorMask, cmap='gray')
        axs[3].set_title("Color Mask", fontsize=8)
        axs[3].axis('off')

        stats_text = f'Length: {l}\n' \
                     f'Width: {w}\n' \
                     f'Color Ratio: {colorRatio:.3f}'
        
        plt.figtext(0.5, 0.05, stats_text, ha='center', fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        # Save the figure
        plt.savefig(save_path, dpi=420, bbox_inches='tight')
        plt.close()

    def hsv_pixel_visualization(self, img, scale=20, font_path=None, save_path='hsv_pixels_labeled.png'):
        from PIL import Image, ImageDraw, ImageFont
        # Load image
        bgr_image = img
        # if bgr_image is None:
        #     raise FileNotFoundError(f"Cannot load image at: {img_path}")

        # Convert to HSV
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        h, w, _ = bgr_image.shape
        scaled_h, scaled_w = h * scale, w * scale

        # Create a new blank RGB image
        out_img = Image.new("RGB", (scaled_w, scaled_h))
        draw = ImageDraw.Draw(out_img)

        # Use default font or provided one
        font_size = 6
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        for y in range(h):
            for x in range(w):
                hsv = hsv_image[y, x]
                bgr = bgr_image[y, x]
                rgb = tuple(int(c) for c in bgr[::-1])  # Convert BGR to RGB for PIL

                top_left = (x * scale, y * scale)
                bottom_right = ((x + 1) * scale, (y + 1) * scale)

                # Draw rectangle with original color
                draw.rectangle([top_left, bottom_right], fill=rgb)

                # Prepare HSV text (scaled to 0-255 range if needed)
                h_val = int(hsv[0] * 2)   # OpenCV uses H in [0,179]
                s_val = int(hsv[1])
                v_val = int(hsv[2])
                text = f"H:{h_val}\nS:{s_val}\nV:{v_val}"

                # Calculate text position
                tx = x * scale + 2
                ty = y * scale + 2

                draw.multiline_text((tx, ty), text, fill=(255, 255, 255), font=font)

        # Save image
        out_img.save(save_path)
        if self.debug:
            logger.info(f"Saved labeled HSV image to {save_path}")

    def processWhite_mask(self, mask, surface=''):        
        mask = mask.astype(np.uint8) * 255

        # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel59, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel59, iterations=1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel5, iterations=2)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel3, iterations=1)

        # mask[:3, :] = 0
        # mask[-3:,:] = 0
        # mask[:,:2]  = 0
        # mask[:,-2:]  = 0

        contours, _   = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            if cv2.contourArea(contour) >= 70:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        # filtered_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel2, iterations=1)
        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_DILATE, self.kernel2, iterations=2)
        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_DILATE, self.kernel3, iterations=1)

        # contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # smooth_mask = np.zeros_like(filtered_mask)
        # for contour in contours:
        #     # Approximate contour with fewer points for smoothing
        #     epsilon = 0.005 * cv2.arcLength(contour, True)
        #     smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        #     # Draw smoothed contour
        #     cv2.drawContours(smooth_mask, [smoothed_contour], -1, 255, -1)

        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_ERODE, self.kernel2, iterations=1)
        
        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, self.kernel2, iterations=1)
        filtered_mask[:5,:]  = 0
        filtered_mask[-5:,:] = 0

        filtered_mask[:,:3]  = 0
        filtered_mask[:,-3:] = 0

        return filtered_mask
    
    def processColor_mask(self, mask, surface='', patchDims=(1,1)):
        mask = mask.astype(np.uint8) * 255
        # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel3, iterations=1)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel3, iterations=1)
        if (patchDims[0] >= patchDims[1]):
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel59, iterations=1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel3, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel59, iterations=1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel5, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel3, iterations=1)
        elif patchDims[1] > patchDims[0]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel95, iterations=1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel3, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel95, iterations=1)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel5, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, self.kernel3, iterations=1)

        # mask[:3, :] = 0
        # mask[-3:,:] = 0
        # mask[:,:2]  = 0
        # mask[:,-2:]  = 0

        mask[:,:5]  = 0
        mask[:,-5:] = 0

        mask[:5,:]  = 0
        mask[-5:,:] = 0

        filtered_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            if self.debug:
                logger.info(f"contour {i}: {cv2.contourArea(contour)}")
            if cv2.contourArea(contour) >= 40: #TODO: add in reconfigurable parameter
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_DILATE, self.kernel3, iterations=1)
        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, self.kernel3, iterations=1)
        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, self.kernel2, iterations=1)
        # filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_ERODE, self.kernel2, iterations=1)

        return filtered_mask

    def binarize_patch(self, patch):
        height, width, _ = patch.shape
        patchgray    = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        thres        = (np.max(patchgray)+np.mean(patchgray))//2
        patchBinary1 = cv2.threshold(patchgray, thres, 255, cv2.THRESH_BINARY)[1]
        
        if height >= width:
            patchBinary = cv2.morphologyEx(patchBinary1, cv2.MORPH_DILATE, self.kernel59, iterations=1)
        elif height < width:
            patchBinary = cv2.morphologyEx(patchBinary1, cv2.MORPH_DILATE, self.kernel95, iterations=1)
        patchBinary  = cv2.morphologyEx(patchBinary, cv2.MORPH_ERODE, self.kernel3, iterations=1)

        # Remove very small area contours from the binary
        contours, _ = cv2.findContours(patchBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_patchBinary = np.zeros_like(patchBinary)
        for contour in contours:
            if cv2.contourArea(contour) >= 30:  # Remove very small area contours (area < 40)
                cv2.drawContours(filtered_patchBinary, [contour], -1, 255, -1)

        if self.debug:
            return patchgray, patchBinary1, filtered_patchBinary
        return filtered_patchBinary
    
    def get_preciseWH(self, patch1, patch2):
        height, width, _ = patch2.shape
        if self.debug:
            patchgray    = None
            patchBinary1 = None
            patchBinary  = None
        # height greater than width case
        if height >= width:
            try:
                # try estimating height and width from patch1 (white mask from processed patch)
                nonzero_y = np.where(patch1.sum(axis=1) > 0)[0]
                top_y     = nonzero_y[0]
                bottom_y  = nonzero_y[-1]
                preciseH  = bottom_y - top_y
                
                # height adjustments (emperical)
                # if preciseH==height:
                #     preciseH = 0.95 * preciseH
                if preciseH < (0.80*height):
                    diff     = height - preciseH
                    offset   = 0.90 * diff
                    preciseH = preciseH + offset

                widths = []
                
                contours, _ = cv2.findContours((patch1 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if self.debug:
                    logger.info(f"Number of contours in patch1: {len(contours)}")
                
                # checking widths contour by contour to avoid high error in width estimation
                for contour in contours:
                    contour_mask = np.zeros_like(patch1)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                    
                    for y in range(top_y, bottom_y + 1):
                        nonzero_x = np.where(contour_mask[y] > 0)[0]
                        if len(nonzero_x) > 0:
                            # x_ranges.append((nonzero_x[0], nonzero_x[-1]))
                            _width = nonzero_x[-1] - nonzero_x[0]
                            _width = _width - 2 if _width>10 else _width
                            widths.append(_width)
                
                widths_array = np.array(widths)
                widths_array[widths_array==0] = 1
                mean = np.mean(widths_array)

                # width adjustments (emperical)
                if mean > 12:
                    diff = mean - 12
                    offset   = 1.4 ** (diff/4)
                    preciseW = mean - offset
                    preciseW = preciseW if preciseW >= 12 else 12
                elif mean < 6:
                    preciseW = 6
                else:
                    preciseW = mean
            except Exception as e:
                logger.error(f"Error in get_preciseWH (height>width; color processing block): {e}")
                # if patch1 is invalid (no contours), try to estimate height and width from patch2 (original patch)
                logger.warning(f"[Warning 1]: Failed to estimate Height/Width from processed white mask, trying to estimate from original patch")
                if self.debug:
                    patchgray, patchBinary1, patchBinary = self.binarize_patch(patch2)
                else:
                    patchBinary = self.binarize_patch(patch2)
                
                try:
                    nonzero_y = np.where(patchBinary.sum(axis=1) > 0)[0]
                    top_y     = nonzero_y[0]
                    bottom_y  = nonzero_y[-1]
                    preciseH  = bottom_y -top_y
                    # height adjustments (emperical)
                    if preciseH==height:
                        preciseH  = 0.95 * preciseH
                    elif preciseH <= (0.75*height):
                        diff     = height - preciseH
                        offset   = 0.8 * diff
                        preciseH = preciseH + offset
                    
                    widths = []
                    
                    contours, _ = cv2.findContours((patchBinary > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if self.debug:
                        logger.info(f"Number of contours in patchBinary: {len(contours)}")
                    # checking widths contour by contour to avoid high error in width estimation
                    for contour in contours:
                        contour_mask = np.zeros_like(patchBinary)
                        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                        
                        for y in range(top_y, bottom_y + 1):
                            nonzero_x = np.where(contour_mask[y] > 0)[0]
                            if len(nonzero_x) > 0:
                                _width = nonzero_x[-1] - nonzero_x[0]
                                _width = _width - 2 if _width>10 else _width
                                widths.append(_width)
                    
                    widths_array = np.array(widths)
                    widths_array[widths_array==0] = 1
                    mean = np.mean(widths_array)
                    # width adjustments (emperical)
                    if mean > 12:
                        diff = mean - 12
                        offset   = 1.45 ** (diff/4)
                        preciseW = mean - offset
                        preciseW = preciseW if preciseW >= 12 else 12
                    elif mean < 6:
                        preciseW = 6
                    else:
                        preciseW = mean
                except Exception as e:
                    logger.error(f"Error in get_preciseWH (height>width; patch processing block): {e}")
                    # if patch2 is invalid (no contours), use patch dims for estimation
                    logger.warning(f"[Warning 2]: Failed to estimate Height/Width from original patch, using patch dims for estimation")
                    preciseH = 0.7 * height
                    mean     = width/4
                    
                    # width adjustments (emperical)
                    if mean > 12:
                        diff = mean - 12
                        offset   = 1.5 ** (diff/4)
                        preciseW = mean - offset
                        preciseW = preciseW if preciseW >= 12 else 12
                    elif mean < 6:
                        preciseW = 6
                    else:
                        preciseW = mean
        
        # width greater than height case
        elif width>height:
            try:
                # try estimating height and width from patch1 (white mask from processed patch)
                nonzero_x = np.where(patch1.sum(axis=0) > 0)[0]
                left_x    = nonzero_x[0]
                right_x   = nonzero_x[-1]
                preciseW  = right_x - left_x
                # width adjustments (emperical)
                # if preciseW == width:
                #     preciseW = 0.95 * width
                if preciseW  < width/2:
                    preciseW = 1.75 * preciseW
                
                heights = []
                
                contours, _ = cv2.findContours((patch1 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if self.debug:
                    logger.info(f"Number of contours in patch1: {len(contours)}")
                # checking heights contour by contour to avoid high error in height estimation
                for contour in contours:
                    contour_mask = np.zeros_like(patch1)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                    
                    for x in range(left_x, right_x + 1):
                        nonzero_y = np.where(contour_mask[:,x] > 0)[0]
                        if len(nonzero_y) > 0:
                            _height = nonzero_y[-1] - nonzero_y[0]
                            _height = _height - 2 if _height>10 else _height
                            heights.append(_height)
                
                heights_array = np.array(heights)
                heights_array[heights_array==0] = 1
                mean = np.mean(heights_array)
                # height adjustments (emperical)
                if mean > 12:
                    diff     = mean - 12
                    offset   = 1.4 ** (diff/4)
                    preciseH = mean - offset
                    preciseH = preciseH if preciseH >= 12 else 12
                elif mean < 6:
                    preciseH = 6
                else:
                    preciseH = mean
            except Exception as e:
                logger.error(f"Error in get_preciseWH (width>height; color processing block): {e}")
                # if patch1 is invalid (no contours), try to estimate height and width from patch2 (original patch)
                logger.warning(f"[Warning 1]: Failed to estimate Height/Width from processed white mask, trying to estimate from original patch")
                if self.debug:
                    patchgray, patchBinary1, patchBinary = self.binarize_patch(patch2)
                else:
                    patchBinary = self.binarize_patch(patch2)
                
                try:
                    nonzero_x = np.where(patchBinary.sum(axis=0) > 0)[0]
                    left_x    = nonzero_x[0]
                    right_x   = nonzero_x[-1]
                    preciseW  = right_x - left_x
                    # width adjustments (emperical)
                    if preciseW == width:
                        preciseW = 0.98 * width
                    if preciseW  < width/2:
                        preciseW = 1.75 * preciseW
                    
                    heights = []
                    
                    contours, _ = cv2.findContours((patchBinary > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if self.debug:
                        logger.info(f"Number of contours in patchBinary: {len(contours)}")
                    # checking heights contour by contour to avoid high error in height estimation
                    for contour in contours:
                        contour_mask = np.zeros_like(patchBinary)
                        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                        
                        for x in range(left_x, right_x + 1):
                            nonzero_y = np.where(contour_mask[:,x] > 0)[0]
                            if len(nonzero_y) > 0:
                                _height = nonzero_y[-1] - nonzero_y[0]
                                _height = _height - 2 if _height>10 else _height
                                heights.append(_height)
                    
                    heights_array = np.array(heights)
                    heights_array[heights_array==0] = 1
                    mean = np.mean(heights_array)
                    # height adjustments (emperical)
                    if mean > 12:
                        diff     = mean - 12
                        offset   = 1.45 ** (diff/4)
                        preciseH = mean - offset
                        preciseH = preciseH if preciseH >= 12 else 12
                    elif mean < 6:
                        preciseH = 6
                    else:
                        preciseH = mean
                except Exception as e:
                    logger.error(f"Error in get_preciseWH (width>height; patch processing block): {e}")
                    # if patch2 is invalid (no contours), use patch dims for estimation
                    logger.warning(f"[Warning 2]: Failed to estimate Height/Width from original patch, using patch dims for estimation")
                    preciseW = 0.7 * width
                    mean     = height/4
                    
                    # height adjustments (emperical)
                    if mean > 12:
                        diff     = mean - 12
                        offset   = 1.5 ** (diff/4)
                        preciseH = mean - offset
                        preciseH = preciseH if preciseH >= 12 else 12
                    elif mean < 6:
                        preciseH = 6
                    else:
                        preciseH = mean
        if self.debug:
            import matplotlib.pyplot as plt

            # directory creation
            parent_dir = 'Analysis'
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            connectorFolder = os.path.join(parent_dir, f'{self.surfaceName}_{self.connectorID}')
            if not os.path.exists(connectorFolder):
                os.makedirs(connectorFolder, exist_ok=True)   
            analysisDir = os.path.join(connectorFolder, 'size_analysis')
            if not os.path.exists(analysisDir):
                os.makedirs(analysisDir, exist_ok=True)

            fig, axs = plt.subplots(1, 5, figsize=(25, 20))
            axs[0].imshow(cv2.cvtColor(patch2, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original Patch", fontsize=14)
            axs[1].imshow(patch1)
            axs[1].set_title("Color Mask", fontsize=14)
            # plt.show()

            if patchgray is not None:
                axs[2].imshow(patchgray, cmap='gray')
                axs[2].set_title("Gray Patch", fontsize=14)
            if patchBinary1 is not None:
                axs[3].imshow(patchBinary1)
                axs[3].set_title("Binary Patch", fontsize=14)
            if patchBinary is not None:
                axs[4].imshow(patchBinary)
                axs[4].set_title("Processed Binary Patch", fontsize=14)
            
            plt.figtext(0.5, -0.06, f'Precise Height pixels: {preciseH:.2f}, Precise Width pixels: {preciseW:.2f}', 
                        ha='center', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
            plt.figtext(0.5, -0.03, f'Precise Height mm: {preciseH/(self.denom+5):.2f}, Precise Width mm: {preciseW/(self.denom+5):.2f}', 
                        ha='center', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
            plt.tight_layout()
            
            # save_path = f'{analysisDir}/size_analysis_{self.surfaceName}_{self.connectorID}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            save_path = os.path.join(analysisDir, f'{self.surfaceName}_{self.connectorID}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg') 
            plt.savefig(save_path, bbox_inches='tight', dpi=320)
            plt.close()
            time.sleep(1)

        return preciseH, preciseW

    def preProcess(self, patch, surface=''):         
                    
        patchHsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        
        hue = patchHsv[:, :, 0]/180
        sat = patchHsv[:, :, 1]/255
        val = patchHsv[:, :, 2]/255

        # Downscale hues in the range 0.14 to 0.17 (multiply with < 1)
        hue[(hue > 0.13) & (hue < 0.17)] *= 0.7

        hue[(hue==0.0)] += 0.02
        hue[(hue > 0.96) & (hue <= 1.0)] = 0.02
        # Up scale hues in the range 0.0 to 0.07 (multiply with > 1)
        hue[(hue > 0.01) & (hue < 0.07)] *= 1.6
        hue[(hue>0.07) & (hue<0.09)]     *= 1.2
        hue[(hue>0.17) & (hue<0.25)]     *= 1.5
        hue = np.clip(hue, 0, 1)
        # up scale the saturation where hue is in the range 0.0 to 0.17 and 0.92 to 1.0
        sat[(hue > 0.01) & (hue < 0.17)] *= 3.0
        sat = np.clip(sat, 0, 1)
        # up scale the value where hue is in the range 0.0 to 0.17 and 0.92 to 1.0
        val[(hue > 0.01) & (hue < 0.17)] *= 5.0
        val[(hue>0.17)&(hue<0.96)&(val<0.8)&(sat>0.1)] *= 0.5
        val = np.clip(val, 0, 1)

        # Convert back to HSV
        patchHsv[:, :, 0] = (hue * 180).astype(np.uint8)
        patchHsv[:, :, 1] = (sat * 255).astype(np.uint8)
        patchHsv[:, :, 2] = (val * 255).astype(np.uint8)
        
        patch = cv2.cvtColor(patchHsv, cv2.COLOR_HSV2BGR)
        patch = cv2.medianBlur(patch, 5)
        patch = cv2.GaussianBlur(patch, (5, 5), 0)
        return patch
    
    def getPinROI(self, pinPatch, surface='', itrs=1, pin_idx=0, kernel=None):
            
        pinPatch[:, :15]  = 0
        pinPatch[:, -15:] = 0
        pinPatch[:20, :]  = 0
        pinPatch[-20:, :] = 0

        pinPatch_hsv    = cv2.cvtColor(pinPatch, cv2.COLOR_BGR2HSV)
        pin_v           = pinPatch_hsv[:, :, 2]
        pin_v[pin_v>50] = 255
        pin_v[pin_v<255]= 0

        if self.debug:
            import copy
            original_pin_mask = copy.deepcopy(pin_v)

        # Find width of pin mask at y=150
        # y1 = 200
        # row = pin_v[y1, :]
        # nonzero_indices = np.where(row > 0)[0]
        # if len(nonzero_indices) > 0:
        #     left_x_1    = nonzero_indices[0]
        #     right_x_1   = nonzero_indices[-1]
        #     pin_width_1 = right_x_1 - left_x_1
        #     if self.debug:
        #         logger.info(f"left_x_1: {left_x_1}, right_x_1: {right_x_1}")
        #         # print(f"pin_width_1: {pin_width_1}")
        # # Find width of pin mask at y=450       
        # y2 = pin_v.shape[0]//2-150
        # row = pin_v[y2, :]
        # nonzero_indices = np.where(row > 0)[0]
        # if len(nonzero_indices) > 0:
        #     left_x_2    = nonzero_indices[0]
        #     right_x_2   = nonzero_indices[-1]
        #     pin_width_2 = right_x_2 - left_x_2
        #     if self.debug:
        #         logger.info(f"left_x_2: {left_x_2}, right_x_2: {right_x_2}")
        #         # print(f"pin_width_2: {pin_width_2}")
        # # Find width of pin mask at y=h-100
        # y3 = pin_v.shape[0]//2+150
        # row = pin_v[y3, :]
        # nonzero_indices = np.where(row > 0)[0]
        # if len(nonzero_indices) > 0:
        #     left_x_3    = nonzero_indices[0]
        #     right_x_3   = nonzero_indices[-1]
        #     pin_width_3 = right_x_3 - left_x_3
        #     if self.debug:
        #         logger.info(f"left_x_3: {left_x_3}, right_x_3: {right_x_3}")
        #         # print(f"pin_width_3: {pin_width_3}")
        # y4 = pin_v.shape[0]//2+280
        # row = pin_v[y4, :]
        # nonzero_indices = np.where(row > 0)[0]
        # if len(nonzero_indices) > 0:
        #     left_x_4    = nonzero_indices[0]
        #     right_x_4   = nonzero_indices[-1]
        #     pin_width_4 = right_x_4 - left_x_4
        
        # if pin_idx in [17,18]:
        #     y5 = pin_v.shape[0]-80
        # else:
        #     y5 = pin_v.shape[0]-40
        # row = pin_v[y5, :]
        # nonzero_indices = np.where(row > 0)[0]
        # if len(nonzero_indices) > 0:
        #     left_x_5    = nonzero_indices[0]
        #     right_x_5   = nonzero_indices[-1]
        #     pin_width_5 = right_x_5 - left_x_5
        #     if self.debug:
        #         logger.info(f"left_x_5: {left_x_5}, right_x_5: {right_x_5}")
        #         # print(f"pin_width_4: {pin_width_4}")
        # try:
        #     if (pin_width_1 > pin_width_5):
        #         left_diff_1 = left_x_5 - left_x_1
        #         if left_diff_1 > 0:
        #             width_diff = pin_width_1 - pin_width_5
        #             corrected_x1 = left_x_1 + width_diff - 4
        #             pin_v[y1:y2, :corrected_x1] = 0
        #             pin_v[:y1, :corrected_x1-1] = 0
        #         right_diff_1 = right_x_1 - right_x_5
        #         if right_diff_1 > 0:
        #             width_diff = pin_width_1 - pin_width_5
        #             corrected_x1 = right_x_1 - width_diff + 4
        #             # pin_v[y1:y2, corrected_x1:] = 0
        #             pin_v[:y1, corrected_x1+1:] = 0
        #             pin_v[y1:y2, corrected_x1:] = 0
        #     if (pin_width_2>pin_width_5):
        #         left_diff_2 = left_x_5 - left_x_2
        #         if left_diff_2 > 0:
        #             width_diff = pin_width_2 - pin_width_5
        #             corrected_x2 = left_x_2 + width_diff -3
        #             pin_v[y2:y3, :corrected_x2] = 0
        #         right_diff_2 = right_x_2 - right_x_5
        #         if right_diff_2 > 0:
        #             width_diff = pin_width_2 - pin_width_5
        #             corrected_x2 = right_x_2 - width_diff + 3
        #             pin_v[y2:y3, corrected_x2:] = 0
        #     if (pin_width_3>pin_width_5):
        #         left_diff_3 = left_x_5 - left_x_3
        #         if left_diff_3 > 0:
        #             width_diff = pin_width_3 - pin_width_5
        #             corrected_x3 = left_x_3 + width_diff -2
        #             pin_v[y3:y4, :corrected_x3] = 0
        #         right_diff_3 = right_x_3 - right_x_5
        #         if right_diff_3 > 0:
        #             width_diff = pin_width_3 - pin_width_5
        #             corrected_x3 = right_x_3 - width_diff +2
        #             pin_v[y3:y4, corrected_x3:] = 0
        #     if (pin_width_4>pin_width_5):
        #         left_diff_4 = left_x_5 - left_x_4
        #         if left_diff_4 > 0:
        #             width_diff = pin_width_4 - pin_width_5
        #             corrected_x4 = left_x_4 + width_diff -1
        #             pin_v[y4:, :corrected_x4] = 0
        #         right_diff_4 = right_x_4 - right_x_5
        #         if right_diff_4 > 0:
        #             width_diff = pin_width_4 - pin_width_5
        #             corrected_x4 = right_x_4 - width_diff +1
        #             pin_v[y4:, corrected_x4:] = 0
                    
        # except Exception as e:
        #     self.logger.error(f"Error in getPinROI: {e}")
 
        # Remove the edges from the top
        y = 200
        row = pin_v[y, :]
        nonzero_indices = np.where(row > 0)[0]
        if len(nonzero_indices) > 0:
            left_x_1    = nonzero_indices[0]
            right_x_1   = nonzero_indices[-1]
            pin_v[:150, :left_x_1-2] = 0
            pin_v[:150, right_x_1+2:] = 0

        pin_v = cv2.morphologyEx(pin_v, cv2.MORPH_CLOSE, self.kernel59, iterations=3)

        # blurred = cv2.GaussianBlur(pin_v, (5, 21), 0)  # (width, height) - height > width for vertical
        # _, pin_v = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # contours, _ = cv2.findContours(pin_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # smoothed_pinV = np.zeros_like(pin_v)

        # for cnt in contours:
        #     if cv2.contourArea(cnt) < 2000:
        #         continue
        #     approx = cv2.approxPolyDP(cnt, epsilon=3, closed=True)
        #     cv2.drawContours(smoothed_pinV, [approx], -1, 255, thickness=cv2.FILLED)

        smoothed_pinV = cv2.morphologyEx(pin_v, cv2.MORPH_ERODE, kernel, iterations=itrs + 1)
        # smoothed_pinV = cv2.morphologyEx(smoothed_pinV, cv2.MORPH_CLOSE, self.kernel4, iterations=1)
        # if not 'Top-pin_auto_0' in surface:
        #     smoothed_pinV = cv2.morphologyEx(smoothed_pinV, cv2.MORPH_ERODE, self.kernel3, iterations=1)
        # elif 'Top-pin_auto_0' in surface and pin_idx in [17,18]:
        #     smoothed_pinV = cv2.morphologyEx(smoothed_pinV, cv2.MORPH_ERODE, self.kernel3, iterations=1)
        smoothed_pinV = cv2.morphologyEx(smoothed_pinV, cv2.MORPH_OPEN, self.kernel59, iterations=3)
        pinPatch = cv2.bitwise_and(pinPatch, pinPatch, mask=smoothed_pinV)
        # if self.debug:
        #     return pinPatch, original_pin_mask, smoothed_pinV
        # else:
        return pinPatch

    def checkBrightness(self, patch):
        # patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        brightness_r   = np.sum(patch[:, :, 2])
        brightness_g   = np.sum(patch[:, :, 1])
        brightness_b   = np.sum(patch[:, :, 0])
        brightness = brightness_r + brightness_g + brightness_b
        height, width = patch.shape[:2]
        avg_brightness = brightness / (3 *width * height)
        return avg_brightness

    def simulate_encoded_image(self, image_np: np.ndarray, format: str = 'png', quality: int = 80) -> np.ndarray:
        """

        Args:
            image_np (np.ndarray): input BGR.
            format (str): 'png' or  'webp'.
            quality (int): 
        Returns:
            np.ndarray: after encode-decode, still in BGR.
        """
        if format == 'png':
            encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # compress            
            ext = '.png'
        elif format == 'webp':
            encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]  # quality webp
            ext = '.webp'
        else:
            raise ValueError("Unsupported format. Use 'png' or 'webp'.")

        success, encoded_image = cv2.imencode(ext, image_np, encode_param)
        if not success:
            logger.error(f"Encoding to {format} failed.")
            return image_np

        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        return decoded_image
    
    def patchify(self, img: np.ndarray, pinPreds=[], surface='', erosion_kernel=[]):
        
        # patches = {'patches':[], 'patchBrightness':[], 'originalDims':[]}
        patches = {'patches':[], 'originalDims':[]}
        for i, pin in enumerate(pinPreds):
            bigPin = False
            if i in [17,18]:
                bigPin = True
            x4, y4, x3, y3, x2, y2, x1, y1 = pin

            patch = np.copy(img[int(y1-5):int(y3+5), int(x1-30):int(x3+30)])
            if 'Top-pin_auto_0' in surface and bigPin:
                bigPin_itrs = self.itrs+3 if self.itrs<4 else 6
                patch = self.getPinROI(patch, surface=surface[0], itrs=bigPin_itrs, pin_idx=i, kernel=erosion_kernel) #TODO: add itrs reconfigurable parameter
            else:
                patch = self.getPinROI(patch, surface=surface[0], itrs=self.itrs, pin_idx=i, kernel=erosion_kernel)
            # brightness = self.checkBrightness(patch)
            # patches['patchBrightness'].append(brightness)
            h,w, _ = patch.shape
            patch_padded = np.zeros((h,w+200,3)).astype(np.uint8)
            patches['originalDims'].append((h,w+200))
            patch_padded[:, 100:w+100, :] = patch
            patch_padded = cv2.resize(patch_padded, (640, 1200), interpolation=cv2.INTER_LINEAR)
            patches['patches'].append(patch_padded)
        return patches

    def processPreds(self, yolo_rslt=[]):
        preds = []
        for i,result in enumerate(yolo_rslt):
            if result.boxes.shape[0] > 0:
                boxes = result.boxes.xyxyn.cpu().numpy()
                conf  = result.boxes.conf.cpu().numpy()
                _preds = np.hstack([boxes, conf[:,None]])
                preds.append(_preds)
            else:
                preds.append(np.array([], dtype=np.float32))
        return preds
    
    def pinPredstoImg(self, preds=[], pinPreds=[], originalDims=[]):
        img_preds = np.empty((1,5))
        for i, pred in enumerate(preds):
            # If there are more than 1 boxes, check for overlaps and consolidate
            if len(pred) > 1:
                pred = self.consolidate_overlapping_boxes(pred)
            pin_pred     = pinPreds[i]
            original_dim = originalDims[i]
            if len(pred) > 0:
                pred[:,[0,2]] *= original_dim[1]
                pred[:,[1,3]] *= original_dim[0]
                pred[:,[0,2]] -= 130
                pred[:,[0,2]] += pin_pred[6]
                pred[:,[1,3]] -= 5
                pred[:,[1,3]] += pin_pred[7]
                img_preds = np.vstack((img_preds, pred))
        return img_preds[1:]
    
    def getDims(self, patch):

        hh, ww, _ = patch.shape

        patch_pp = self.preProcess(patch)
        patchHSV = cv2.cvtColor(patch_pp, cv2.COLOR_BGR2HSV)
        
        # color mask estimation
        redMask    = self.filterRed(patchHSV)
        yellowMask = self.filterYellow(patchHSV)
        orangeMask = self.filterOrange(patchHSV)
        brownMask  = self.filterBrown(patchHSV)
        copperMask = self.filterCopper(patchHSV)
        # whiteMask  = self.filterWhite(patchHSV)

        colorMask_raw = redMask | yellowMask | orangeMask | brownMask | copperMask

        # whiteMask = self.processWhite_mask(whiteMask)
        colorMask = self.processColor_mask(colorMask_raw, (hh,ww))

        h, w = self.get_preciseWH(colorMask, patch)
        
        colorPixels = cv2.countNonZero(colorMask)
        colorRatio = colorPixels / ((h*w)+100)
        colorRatio = colorRatio if colorRatio <= 1.0 else 1.0
        
        if self.debug:
            import matplotlib.pyplot as plt
            
            parent_dir = 'Analysis'
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            connectorFolder = os.path.join(parent_dir, f'{self.surfaceName}_{self.connectorID}')
            if not os.path.exists(connectorFolder):
                os.makedirs(connectorFolder, exist_ok=True)   
            analysisDir = os.path.join(connectorFolder, 'color_analysis')
            if not os.path.exists(analysisDir):
                os.makedirs(analysisDir, exist_ok=True)
            
            fig, axs = plt.subplots(1, 4, figsize=(25, 20))
            axs[0].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            axs[0].set_title("Original Patch", fontsize=14)
            axs[1].imshow(cv2.cvtColor(patch_pp, cv2.COLOR_BGR2RGB))
            axs[1].set_title("Preprocessed Patch", fontsize=14)
            # axs[2].imshow(whiteMask)
            # axs[2].set_title("White Mask", fontsize=14)
            axs[2].imshow(colorMask_raw.astype(np.uint8)*255)
            axs[2].set_title("Original Color Mask", fontsize=14)
            
            axs[3].imshow(colorMask)
            axs[3].set_title("ProcessedColor Mask", fontsize=14)

            plt.figtext(0.5, -0.02, f'Color Pixels: {colorRatio:.3f}', 
                        ha='center', fontsize=18, bbox=dict(facecolor='white', alpha=0.8))
            # plt.show()
            # save_path = f'Analysis/patch_analysis_{self.surfaceName}_{self.connectorID}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
            save_path = os.path.join(analysisDir, f'{self.surfaceName}_{self.connectorID}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg') 
            plt.savefig(save_path, bbox_inches='tight', dpi=320)
            plt.close()
            time.sleep(1)

        return max(h,w), min(h,w), colorRatio
    
    def postProcess(self, img:np.ndarray, preds=[]):
        rslts = {'selectedPreds':[], 'discardedPreds':[]}
        for i, pred in enumerate(preds):
            x1, y1, x2, y2, conf = pred
            patch = np.copy(img[int(y1):int(y2), int(x1):int(x2)])

            if patch.shape[0] > 20:
                patch[:5,:,:]  = 0
                patch[-5:,:,:] = 0
            else:
                patch[:2,:,:]  = 0
                patch[-2:,:,:] = 0
            if patch.shape[1] > 20:    
                patch[:,:5,:]  = 0
                patch[:,-5:,:] = 0   
            else:
                patch[:,:2,:]  = 0
                patch[:,-2:,:] = 0
            # if self.debug:
            #     l,w,colorRatio, patch_pp, whiteMask, colorMask = self.getDims(patch)
            #     self.visualize(patch, patch_pp, whiteMask, colorMask, l, w, colorRatio, save_path=f'Analysis/patch_{i}.jpg')
            # else:
            l,w,colorRatio = self.getDims(patch)

            l_mm = l/(self.denom + 5)
            w_mm = w/(self.denom + 5)
            
            area_mm = l_mm * w_mm

            if area_mm > self.anom_area:
                _pred = np.hstack((pred, l_mm, w_mm, colorRatio))
                rslts['selectedPreds'].append(_pred)
            else:
                _pred = np.hstack((pred, l_mm, w_mm, colorRatio))
                rslts['discardedPreds'].append(_pred)
        return rslts
    
    def consolidate_overlapping_boxes(self, pred):
        N = len(pred)
        boxes = pred[:, :4]
        
        # Compute IoU matrix using broadcasting
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Reshape for broadcasting
        x1_i, x1_j = x1[:, None], x1[None, :]
        y1_i, y1_j = y1[:, None], y1[None, :]
        x2_i, x2_j = x2[:, None], x2[None, :]
        y2_i, y2_j = y2[:, None], y2[None, :]
        
        # Compute intersection coordinates
        inter_xmin = np.maximum(x1_i, x1_j)
        inter_ymin = np.maximum(y1_i, y1_j)
        inter_xmax = np.minimum(x2_i, x2_j)
        inter_ymax = np.minimum(y2_i, y2_j)
        
        # Compute intersection area
        inter_w = np.maximum(0, inter_xmax - inter_xmin)
        inter_h = np.maximum(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Compute union area
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_j = (x2_j - x1_j) * (y2_j - y1_j)
        union_area = area_i + area_j - inter_area
        
        # Compute IoU
        iou = np.where(union_area > 0, inter_area / union_area, 0)
        
        # Create overlap mask (IoU > 0.5, excluding self)
        overlap_mask = (iou > 0.4) & ~np.eye(N, dtype=bool)
        
        # Group overlapping boxes
        processed = np.zeros(N, dtype=bool)
        merged = []
        
        for i in range(N):
            if processed[i]:
                continue
            
            # Find all boxes that overlap with current box
            overlapping_indices = np.where(overlap_mask[i])[0]
            group_indices = [i] + overlapping_indices.tolist()
            
            # Mark all boxes in group as processed
            processed[group_indices] = True
            
            # Consolidate the group
            group_boxes = pred[group_indices]
            x_min = np.min(group_boxes[:, 0])
            y_min = np.min(group_boxes[:, 1])
            x_max = np.max(group_boxes[:, 2])
            y_max = np.max(group_boxes[:, 3])
            max_conf = np.max(group_boxes[:, 4])
            
            merged.append([x_min, y_min, x_max, y_max, max_conf])
        
        return np.array(merged)
    
    def process(self, image: np.ndarray, previous_detections = [], info={}, updated_parameters={}):  #TODO 7: You will receive image and the detection results from previous model (if this model is not the first one)
        """
        Run inference on the provided image using the pre-loaded YOLO model.

        Args:
            image (np.ndarray): The input image array.
            previous_detections (list): A list of detection results from the previous model.

        Returns:
            annotated_image (np.ndarray): The image annotated with detection results.
            detections (list): A list of detection results, each containing bounding boxes,
                                confidence scores, classes, and labels.
        """

        try:
            logger.info(" ")
            logger.info("m08_Abrasion Model")
            logger.info(f"Surface Name: {info['Input']}")
            logger.info(f"Image Acquisition Time: {datetime.datetime.now()}")    
            # fst = time.time()  
            # Check if the input image is valid
            if image is None:
                self.logger.error("Input image is empty or None.")
                raise ValueError("Input image is empty or None.")
            # st = time.time()
            try:
                self.conf_thres = updated_parameters["confidence_threshold"]
                logger.info(f"[confidence_threshold Update] updated value: {self.conf_thres}")
                
                self.annotate_SVG = updated_parameters["SVG_annotation"]
                logger.info(f"[SVG_Annotation Update] updated value: {self.annotate_SVG}")

                self.showFiltered = updated_parameters["show_blue_boxes"]
                logger.info(f"[show_blue_boxes Update] updated value: {self.showFiltered}")

                self.itrs = int(updated_parameters["erosion_iterations"])
                logger.info(f"[erosion_iterations Update] updated value: {self.itrs}")

                self.erosion_kernel = int(updated_parameters["erosion_kernel"])
                logger.info(f"[erosion_kernel Update] updated value: {self.erosion_kernel}")

                self.anom_width = updated_parameters["anomaly_width"]
                logger.info(f"[anomaly_width Update] updated value: {self.anom_width}")

                self.anom_height = updated_parameters["anomaly_height"]
                logger.info(f"[anomaly_height Update] updated value: {self.anom_height}")
                
                self.denom = int(updated_parameters["dividing_factor"])
                logger.info(f"[denom Update] updated value: {self.denom}")

            except Exception as e:
                logger.error(f"[Update param] Error: {e}") 
                # pass
            # logger.info(f"[Get Update param] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
            
            if self.erosion_kernel == 9:
                erosion_kernel = self.kernel9
            elif self.erosion_kernel == 7:
                erosion_kernel = self.kernel7
            elif self.erosion_kernel == 5:
                erosion_kernel = self.kernel5
            else:
                logger.warning("Wrong Kernel Selected, Choosing Default Kernel")
                erosion_kernel = self.kernel5            
            self.anom_area = self.anom_width * self.anom_height
            # start_time = time.time()
            # st = time.time()            
            # image = self.simulate_encoded_image(image, format='webp', quality=100)
            img              = np.copy(image)
            height, width, _ = img.shape
            roiRslts         = self.roiModel.process(img, info=info, call_from='Abrasion_model', roiExtract=False)
            # self.logger.info(f"[ROI Model] Time taken: {(time.time() - st) * 1000:.4f} milliseconds")
            
            img = self.simulate_encoded_image(img, format='png')

            # svg_annots        = None
            # anomaly_result    = False
            # detection_results = []

            if isinstance(roiRslts, dict):
                # st = time.time()
                pinPreds   = roiRslts['upperPinsObb_preds']
                pinPreds[:, [0,2,4,6]] /= 1920  # Normalize x coordinates
                pinPreds[:, [0,2,4,6]] *= width
                pinPreds[:, [1,3,5,7]] /= 1280  # Normalize y coordinates
                pinPreds[:, [1,3,5,7]] *= height

                surface = roiRslts['Surface_names']
                roiBox  = np.array(roiRslts['pinROI_box']).astype(np.float64)
                roiBox[::2]  /= 1920  # x coordinates
                roiBox[1::2] /= 1280  # y coordinates
                roiBox[::2]  *= width
                roiBox[1::2] *= height
                
                roiBox[1] += 20
                roiBox[3] -= 200
                self.surfaceName = surface[0]
                self.connectorID = info.get('Connector_ID')

                sorted_indices = np.argsort(pinPreds[:, 6])
                pinPreds       = pinPreds[sorted_indices]

                if 'Top' in surface[0]:
                    pos1 = (60,120)
                    pos2 = (300,120)
                    pos3 = (60, 220)
                    y_limit = 2700
                    # if 'Top-pin_auto_0' in surface[0]:
                    #     pin18 = pinPreds[17]
                    #     self.fpROI1 = [int(pin18[6]-50), int(pin18[7]+20), int(roiBox[2]-10), int(pin18[7]+60)]
                    # self.surfaceConf = self.conf_thres_top

                # if 'Top-pin_auto_0' in surface[0]:
                    
                #     # pinPreds[-2:, [2, 4]] -= 8
                #     # pinPreds[-2:, [0, 6]] += 4
                #     pinPreds[-2:, [1, 3]] -= 210
                #     pinPreds[:-2, [1, 3]] -= 200
                # elif 'Top-pin_auto_1' in surface[0]:
                #     pinPreds[:, [1,3]] -= 200
                pinPreds[:, [1,3]] -= 200
                
                patches = self.patchify(img, pinPreds, surface[0], erosion_kernel=erosion_kernel)
                _patches, originalDims = patches.get('patches'), patches.get('originalDims')

                return _patches
        except Exception as e:
            logger.error(f"[Abrasion Model] Error: {e}") 

# Example usage:
if __name__ == "__main__":
    import tqdm
    from natsort import natsorted
    # import cairosvg
    from PIL import Image
    import io
    # Load the model
    # roimodel = roiModel(model_path='m07_pinROI.pt', confidence_threshold=0.25, warmup_runs=3)
    # model_path = "/AIRobot/best-04-22.pt"
    roiModel_path = "/home/zafar/membrane_crack_Anom/m07_pinROI.pt"
    roiModel_confidence = 0.2
    abr_model_path = "/home/zafar/membrane_crack_Anom/abr_best_3_15.pt"
    abr_model_conf = 0.3

    abrmodel = MODEL(model_path=abr_model_path, 
                     confidence_threshold=abr_model_conf, 
                     roiModel_path=roiModel_path,
                     roiModel_confidence=roiModel_confidence)

    dataPath = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/20250703-D/top_1.txt'
    save_path = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/new_data_factory_bldng/abr_data_D_2'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with open(dataPath, 'r') as f:
        imgPaths = f.readlines()
    imgPaths = natsorted(imgPaths)
    # svg = True
    # Load an image
    imageFormats = ['.png', '.jpg', '.jpeg', '.bmp', '.webp']
    idx = 0
    for imagePath in tqdm.tqdm(imgPaths):
        # /root/Desktop/AIRobot/20250523-A-20/results_abr/00002_Top-pin_auto_1_ModifiedExposure.png
        # imagePath = "/root/Desktop/AIRobot/20250617-A/Top-pin_auto_0/00013_Top-pin_auto_0.png"
        print(imagePath)
        imagePath    = imagePath.strip()
        originalName = imagePath.split('/')[-1]
        if not (originalName.endswith('.png') or originalName.endswith('.jpg')):
            originalName = originalName.split('.')[0] + '.png'
        # surfaceName  = imagePath.split('Input-')[1].split('__Cam')[0]
        surfaceName  = imagePath.split('/')[-2]
        id = originalName.split('_')[0]
        # surfaceName  = 'Top-pin_auto_0'
        info={'Input':surfaceName, 'Connector_ID':id}

        ext = os.path.splitext(imagePath)[1].lower()
        if ext in imageFormats:
            image = cv2.imread(imagePath)
        elif ext=='.npy':
            image = np.load(imagePath)
        else:
            raise NotImplementedError(f"Unsupported image format: {ext}")
        
        patches =abrmodel.process(image=image, info=info)

        for patch in patches:
            name = surfaceName + '_' + str(idx) + '.jpg'
            path = os.path.join(save_path, name)
            cv2.imwrite(path, patch)
            idx += 1