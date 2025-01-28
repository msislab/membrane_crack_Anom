import cv2, os, argparse, tqdm, glob
import numpy as np
from natsort import natsorted
from ultralytics import YOLO
import torch
from shapely.geometry import Polygon
import time
import colorama
from colorama import Style, Fore, Back
colorama.init(autoreset=True)


def parseArgs():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide img directory path')
    parser.add_argument('--model', type=str, default=None,
                        help='Provide model.pt path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--iou_thres', type=float, default=0.30,
                        help='Choose an iou threshold between 0 and 1')
    parser.add_argument('--PredictYolo', action='store_true',
                        help='to choose prediction and visulization mode')
    parser.add_argument('--PredYolo_obb', action='store_true',
                        help='to choose prediction and visulization mode')
    parser.add_argument('--multiLevel', action='store_true',
                        help='to enable multi-view prediction')
    parser.add_argument('--visualizeGT', action='store_true',
                        help='To choose visualize only mode')
    parser.add_argument('--obbGT', action='store_true',
                        help='To enable visualization of obb annotation')
    parser.add_argument('--output', type=str, default='output',
                        help='Specify a directory to save outputs')
    parser.add_argument('--save', action='store_true',
                        help='Choose save if you want to save the visualization results')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose a gpu device')
    parser.add_argument('--boxType', type=str, default='xywh',
                        help='Choose box type for annotation')
    parser.add_argument('--imgSize', type=int, default=640,
                        help='Choose box type for annotation')
    args = parser.parse_args()
    return args

def calculate_iou(box1, box2):
    poly1 = Polygon(box1.reshape(4, 2))
    poly2 = Polygon(box2.reshape(4, 2))

    # if not poly1.is_valid or not poly2.is_valid:
    #     return 0.0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    return intersection_area / union_area if union_area > 0 else 0.0

def find_overlaps(gt_boxes, pred_boxes, iou_threshold=0.5, stats=False):
    overlaps  = np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]))

    matched = set()  # Tracks matched ground truth indices
    tp, fp = 0, 0

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_idx = -1
        pred_class, pred_coords = pred_box[0], pred_box[1:9]
        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_class, gt_coords = gt_box[0], gt_box[1:]

            # in case of more than 1 classes
            # if gt_class != pred_class:
            #     continue
            if gt_idx not in matched:
                iou = calculate_iou(gt_coords, pred_coords)
                if iou > best_iou:
                    overlaps[gt_idx, pred_idx] = iou
                    best_iou = iou
                    best_idx = gt_idx
        if best_iou >= iou_threshold:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1  
    
    if stats:
        # Get indices of the maximum value in each row
        # max_indices = np.argmax(overlaps, axis=1)
        # Create an output array of zeros with the same shape
        # _overlaps = np.zeros_like(overlaps)
        # Set the maximum values in the corresponding positions
        # _overlaps[np.arange(overlaps.shape[0]), max_indices] = overlaps[np.arange(overlaps.shape[0]), max_indices]
        # return _overlaps
        return tp, fp
    else:
        return overlaps

def stats_obb(_Path, preds, iou_thres):
    img = cv2.imread(_Path)
    h,w,_ = img.shape
    labelPath = _Path.split('.jpg')[0] + '.txt'
    boxes = []
    with open(labelPath, 'r') as f:
        for line in f.readlines():
            boxStr = line.split()
            box = [int(boxStr[0]),float(boxStr[1])*w, float(boxStr[2])*h, 
                    float(boxStr[3])*w, float(boxStr[4])*h,
                    float(boxStr[5])*w, float(boxStr[6])*h,
                    float(boxStr[7])*w, float(boxStr[8])*h]
            boxes.append(np.array(box))
    boxes  = np.array(boxes)
    tp, fp = find_overlaps(boxes, preds, stats=True)
    # Fn_GT     = np.where(np.all(overlaps_ == 0, axis=1))[0]
    # Fp_preds  = np.where(np.all(overlaps_ < iou_thres, axis=0))[0]

    total_Gt    = boxes.shape[0]
    total_preds = preds.shape[0]
    # tp = len(overlaps_[overlaps_>=iou_thres])
    # fp = total_preds-tp
    # print(total_preds, '\n', tp, '\n', fp)
    # time.sleep(2)
    return total_Gt, total_preds, tp, fp

def drawBox(img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1):
    if obb:
        # print('visualizing obb preds')
        for box in boxes:
            _box = np.array([[int(box[1]), int(box[2])],
                             [int(box[3]), int(box[4])],
                             [int(box[5]), int(box[6])],
                             [int(box[7]), int(box[8])]])
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
    return img    

def visPreds(GTpath=None, img=None, preds=None, obb=False, boxType='xywh', color=(0,0,255)):
    if GTpath:  # to visualize GT boxes for comparison
        labelPath = GTpath.split('.jpg')[0] + '.txt'
        if obb: # to visualize obb (oriented box with 4 points) preds
            boxes = []
            with open(labelPath, 'r') as f:
                for line in f.readlines():
                    boxStr = line.split()
                    box = [int(boxStr[0]),float(boxStr[1])*img.shape[1], float(boxStr[2])*img.shape[0], 
                            float(boxStr[3])*img.shape[1], float(boxStr[4])*img.shape[0],
                            float(boxStr[5])*img.shape[1], float(boxStr[6])*img.shape[0],
                            float(boxStr[7])*img.shape[1], float(boxStr[8])*img.shape[0]]
                    boxes.append(np.array(box))
            boxes = np.array(boxes)
            _img  = drawBox(img, boxes, obb=obb, thickness=1)
        else:   # to visualize box (rectangular box) preds
            boxes = []
            with open(labelPath, 'r') as f:
                for line in f.readlines():
                    boxStr = line.split()
                    box = [int(boxStr[0]),float(boxStr[1])*img.shape[1], float(boxStr[2])*img.shape[0], 
                            float(boxStr[3])*img.shape[1], float(boxStr[4])*img.shape[0]]
                    boxes.append(np.array(box))
            boxes = np.array(boxes)
            _img  = drawBox(img, boxes, boxType=boxType)
    else:       # to visualize pred boxes
        if obb:
            _img  = drawBox(img, preds, obb=obb, color=color, thickness=2)
        else:
            _img  = drawBox(img, preds, boxType=boxType, color=color)
    
    # cv2.imshow('',_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return _img            

def visGT(path=None, obb=False, boxType='xywh'):
    _img = cv2.imread(path)
    labelPath = path.split('.jpg')[0] + '.txt'
    if os.path.isfile(labelPath):
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

            # cv2.imshow('',_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
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

def predyolo(img, args, model):
    # resize image if required
    if img.shape[0]>1080:
        _img = cv2.resize(img, (1920, 1080), cv2.INTER_AREA)
    else: _img=img
    # get model results
    results = model(_img, conf=args.conf, verbose=False)
    # extract yolo preds
    for result in results:
        boxes = result.boxes.xywh.cpu().numpy()
        clsz  = result.boxes.cls.cpu().numpy()
        conf  = result.boxes.conf.cpu().numpy()
    # format preds for visualization
    preds = np.zeros((boxes.shape[0], 6))
    preds[:,0]   = clsz
    preds[:,-1]  = conf
    preds[:,1:5] = boxes
    # visulize predictions
    annotImg = visPreds(img=_img, preds=preds)
    # add ground truth visulizations on same image for visual comparison
    if args.visGT:
        annotImg = visPreds(GTpath=args.path, img=annotImg)
    return annotImg        

def processOBB(yoloOBB_rslt, conf_thres):
    preds = np.zeros((1,10))
    for result in yoloOBB_rslt:
        boxes  = result.obb.xyxyxyxy.cpu().numpy().reshape(-1,8)
        clsz   = result.obb.cls.cpu().numpy()
        conf   = result.obb.conf.cpu().numpy()
        _preds = np.hstack([clsz[:,None], boxes, conf[:,None]])
        preds  = np.vstack((preds, _preds))

    # preds = np.delete(preds, 0, axis=0)
    return preds[preds[:, -1] >= conf_thres]

def consolidate(pred1, pred2):
    _pred1 = np.delete(pred1, -1, axis=1)
    _pred2 = np.delete(pred2, -1, axis=1)
    overlaps   = find_overlaps(_pred1, _pred2)
    overlapped = np.where(overlaps>0.45)
    # Select rows based on overlap condition
    pred1_overlap = pred1[overlapped[0]]
    pred2_overlap = pred2[overlapped[1]]
    # Take predictions with higher confidence
    max_conf_mask = pred1_overlap[:, -1] > pred2_overlap[:, -1]
    preds = np.where(max_conf_mask[:, None], pred1_overlap, pred2_overlap)
    
    _pred1 = np.delete(pred1, overlapped[0], axis=0)
    _pred2 = np.delete(pred2, overlapped[1], axis=0)
    preds  = np.vstack((preds, _pred1, _pred2))
    return preds

def predYOLO_obb(img, args, model, device='cpu'):  
    if args.multiLevel:
        results_1 = model.predict(img, imgsz=(1024,1024), conf=args.conf, device=device, verbose=False)
        results_2 = model.predict(img, imgsz=(640,640), conf=args.conf, device=device, verbose=False)
        preds_1   = processOBB(results_1, args.conf)
        preds_2   = processOBB(results_2, args.conf)
        preds     = consolidate(preds_1, preds_2)
        preds     = preds[preds[:,-1]>0.03]
    else:
        results = model.predict(img, imgsz=args.imgSize, conf=args.conf, device=device, verbose=True)
        preds   = processOBB(results, args.conf)
        # preds   = preds[preds[:,-1]>0.015]

    return preds            

def main():
    args  = parseArgs()
    imgs  = glob.glob(f'{args.dataPath}/*.jpg')
    imgs  = natsorted(imgs)
    stats = False

    if args.save:
        save_dir = f'{args.output}/bur-model-val'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if args.PredictYolo or args.PredYolo_obb:
        args.visGT = True
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        model  = YOLO(args.model)
        # model.to(device)

    if args.PredYolo_obb:
        TP = 0
        FP = 0
        Total_GT   = 0
        Total_Pred = 0
    
    total_time = 0

    for imgPath in tqdm.tqdm(imgs):
        args.path = imgPath
        print(imgPath)
        if args.save:
            name      = imgPath.split('/')[-1]
            saveName  = os.path.join(save_dir, name)

        _img    = cv2.imread(imgPath)

        # cv2.imshow('', _img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        if args.PredictYolo:   
            annotImg = predyolo(img=_img, args=args, model=model)
        elif args.PredYolo_obb:
            s = time.time()
            preds = predYOLO_obb(img=_img, args=args, model=model, device=device)
            e = time.time()
            total_time+=e-s
            # annotate image with preds
            annotImg = visPreds(img=_img, preds=preds, obb=True)
            # annotate with GT
            if args.visGT:
                annotImg = visPreds(GTpath=args.path, img=annotImg, obb=True)
        
            # prediction statistics (validation)
            stats = True
            total_Gt, total_preds, tp, fp = stats_obb(args.path, preds, args.iou_thres)
            TP += tp
            FP += fp
            Total_GT   += total_Gt
            Total_Pred += total_preds                    
        elif args.visualizeGT:
            if args.obbGT:
                annotImg = visGT(imgPath, obb=args.obbGT)
                # cv2.imshow('',annotImg)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
            else:
                annotImg = visGT(imgPath, boxType=args.boxType)
                cv2.imshow('',annotImg)
                cv2.waitKey()
                cv2.destroyAllWindows()

        if args.save:
            cv2.imwrite(saveName, annotImg)

    if stats:
        print('Total inference time is: ', total_time/len(imgs))
        FN        = Total_GT-TP
        precision = round((TP/(TP+FP))*100, 4)
        recal     = round((TP/(TP+FN))*100,4)
        acc       = round((TP/(Total_GT))*100, 4)

        s = ('%20s' + '%13s' * 6) % ('Anomaly', 'Total-GT', 'Total-Preds', 'TP', 'FP', 'P', 'R')
        print(f'{Style.BRIGHT}{s}')
        results = ('%20s' + '%13s' * 6) % ('TML BUR', Total_GT, Total_Pred, TP, FP, precision, recal)
        print(results)
    print()             

if __name__=='__main__':
    main()