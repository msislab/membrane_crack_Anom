import cv2, os, argparse, tqdm, glob
import numpy as np
from natsort import natsorted


def parseArgs():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--dataPath', type=str, default='',
                        help='Provide img directory path')
    parser.add_argument('--model', type=str, default='',
                        help='Provide model.pt path')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Choose a confidence threshold between 0 and 1')
    parser.add_argument('--Predict', action='store_true',
                        help='to choose prediction and visulization mode')
    parser.add_argument('--visualize', action='store_true',
                        help='To choose visualize only mode')
    args = parser.parse_args()
    return args

def drawBox(img, boxes, boxType='xywh', color=(0,255,0), thickness=1):
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
        print(NotImplemented)
    return img    

def vis(img, preds=None):
    if preds:
        pass    # TODO: impliment for prediction visualization
    else:
        _img = cv2.imread(img)
        labelPath = img.split('.jpg')[0] + '.txt'
        boxes = []
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

def main():
    args = parseArgs()
    imgs = glob.glob(f'{args.dataPath}/*.jpg')
    imgs = natsorted(imgs)

    for img in tqdm.tqdm(imgs):
        if args.Predict:
            print(NotImplemented)
        elif args.visualize:
            vis(img)    

if __name__=='__main__':
    main()