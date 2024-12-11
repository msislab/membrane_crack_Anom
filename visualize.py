import cv2, os, argparse, tqdm, glob
import numpy as np
from natsort import natsorted
from ultralytics import YOLO
import torch


def parseArgs():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide img directory path')
    parser.add_argument('--model', type=str, default=None,
                        help='Provide model.pt path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--PredictYolo', action='store_true',
                        help='to choose prediction and visulization mode')
    parser.add_argument('--PredYolo_obb', action='store_true',
                        help='to choose prediction and visulization mode')
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
    args = parser.parse_args()
    return args

def drawBox(img, boxes, obb=False, boxType='xywh', color=(0,255,0), thickness=1):
    if obb:
        print('visualizing obb preds')
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

def visPreds(GTpath=None, img=None, preds=None, obb=False, boxType='xywh'):
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
            _img  = drawBox(img, preds, obb=obb, color=(0, 0, 255), thickness=2)
        else:
            _img  = drawBox(img, preds, boxType=boxType, color=(0, 0, 255))
    
    cv2.imshow('',_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return _img            

def visGT(path=None, obb=False, boxType='xywh'):
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
        _img  = drawBox(_img, boxes, obb=obb, thickness=2)
    else:
        with open(labelPath, 'r') as f:
            for line in f.readlines():
                boxStr = line.split()
                box = [int(boxStr[0]),float(boxStr[1])*_img.shape[1], float(boxStr[2])*_img.shape[0], 
                        float(boxStr[3])*_img.shape[1], float(boxStr[4])*_img.shape[0]]
                boxes.append(np.array(box))
        boxes = np.array(boxes)        
        _img  = drawBox(_img, boxes)

    cv2.imshow('',_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return _img 

def predyolo(img, args, model):
    # resize image if required
    if img.shape[0]>1080:
        _img = cv2.resize(img, (1920, 1080), cv2.INTER_AREA)
    else: _img=img
    # get model results
    results = model(_img, conf=args.conf, verbose=True)
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

def predYOLO_obb(img, args, model):
    if img.shape[0]>1080:
        _img = cv2.resize(img, (1920, 1080), cv2.INTER_AREA)
    else: _img=img    
    results = model(_img, conf=args.conf, verbose=True)
    for result in results:
        boxes  = result.obb.xyxyxyxy.cpu().numpy()
        _boxes = boxes.reshape(boxes.shape[0], -1)
        clsz   = result.obb.cls.cpu().numpy()
        conf   = result.obb.conf.cpu().numpy()
    preds = np.zeros((boxes.shape[0], 10))
    preds[:,0]   = clsz
    preds[:,-1]  = conf
    preds[:,1:9] = _boxes
    annotImg = visPreds(img=_img, preds=preds, obb=True)
    if args.visGT:
        annotImg = visPreds(GTpath=args.path, img=annotImg, obb=True)
    
    # impliment the analysis here
    return annotImg            

def main():
    args = parseArgs()
    imgs = glob.glob(f'{args.dataPath}/*.jpg')
    imgs = natsorted(imgs)

    if args.save:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    if args.PredictYolo or args.PredYolo_obb:
        args.visGT = True
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        model  = YOLO(args.model)
        # model.to(device)

    for imgPath in tqdm.tqdm(imgs):
        args.path = imgPath
        if args.save:
            name      = imgPath.split('/')[-1]
            saveName  = os.path.join(args.output, name)

        _img    = cv2.imread(imgPath)

        cv2.imshow('', _img)
        cv2.waitKey()
        cv2.destroyAllWindows()

        if args.PredictYolo:   
            annotImg = predyolo(img=_img, args=args, model=model)
        elif args.PredYolo_obb:
            annotImg = predYOLO_obb(img=_img, args=args, model=model)                    
        elif args.visualizeGT:
            if args.obbGT:
                annotImg = visGT(imgPath, obb=args.obbGT)
            else:
                annotImg = visGT(imgPath, boxType=args.boxType)

        if args.save:
            cv2.imwrite(saveName, annotImg)         

if __name__=='__main__':
    main()