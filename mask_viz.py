import cv2, glob, os
import numpy as np
import json
import copy

def getTopleft(labels):
    boxes = np.array([label[1] for label in labels])
    topL = (int(min(boxes[:,0])-15), int(min(boxes[:,1])-15))
    return topL

def getBotomright(labels):
    boxes  = np.array([label[1] for label in labels])
    _boxes = np.zeros_like(boxes)
    _boxes[:,0] = boxes[:,0]
    _boxes[:,1] = boxes[:,1]
    _boxes[:,2] = boxes[:,0] + boxes[:,2]
    _boxes[:,3] = boxes[:,1] + boxes[:,3]
    botomR = (int(np.max(_boxes[:,2])+15), int(max(_boxes[:,3])+15))
    return botomR

def makePinImg(labels, img):
    boxes  = np.array([label[1] for label in labels if label[0]==2])
    _boxes = np.zeros_like(boxes)
    _boxes[:,0] = boxes[:,0]
    _boxes[:,1] = boxes[:,1]
    _boxes[:,2] = boxes[:,0] + boxes[:,2]
    _boxes[:,3] = boxes[:,1] + boxes[:,3]
    pinImg = np.zeros_like(img, dtype=np.uint8)
    for box in _boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        pinImg[y1:y2, x1:x2] = img[y1:y2, x1:x2]

    # cv2.imshow('', pinImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return pinImg

def makeBurImg(labels, img):
    boxes  = np.array([label[1] for label in labels if label[0]==2])
    _boxes = np.zeros_like(boxes)
    _boxes[:,0] = boxes[:,0]
    _boxes[:,1] = boxes[:,1]
    _boxes[:,2] = boxes[:,0] + boxes[:,2]
    _boxes[:,3] = boxes[:,1] + boxes[:,3]
    burImg = copy.deepcopy(img)
    for box in _boxes:
        x1, y1, x2, y2 = int(box[0]+2), int(box[1]), int(box[2]-2), int(box[3])
        burImg[y1:y2, x1:x2] = 0

    # cv2.imshow('', pinImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return burImg     
        
    


_path = '/home/zafar/old_pc/data_sets/robot-project-datasets/TML-Pin-detection-v2-tml-pin-detection-coco-segmentation/train'
annot = '/home/zafar/old_pc/data_sets/robot-project-datasets/TML-Pin-detection-v2-tml-pin-detection-coco-segmentation/train/_annotations.json'
savePath = '/home/zafar/old_pc/data_sets/robot-project-datasets'
# files = glob.glob(f'{path}/*.jpg')

# for file in files:
#     img = cv2.imread(file)

#     labelPath = file.split('.jpg')[0] + '.json'

#     with open(labelPath, 'r') as f:
#         annotations = json.load(f)
#     print()
with open(annot, 'r+') as f:
    annotations = json.load(f)
data = [(item['file_name'], item['id']) for item in annotations['images'] ]
for imgName, imgId in data:
    imgPath = os.path.join(_path, imgName)
    img = cv2.imread(imgPath)

    if img is None:
        print(f"Image {imgName} not found. Skipping...")
        continue

    labels = [
        [ann['category_id'], ann['bbox'], ann['segmentation'][0]]
        for ann in annotations['annotations']
        if ann['image_id'] == imgId and ann['category_id'] in [2, 3]
    ]
    categories = np.array([label[0] for label in labels])
    if 3 in categories:
        topL1  = getTopleft(labels)
        _label = labels[int(np.where(categories==3)[0])]
        box    = _label[1]
        topL2  = (int(box[0]), int(box[1]+30))
        botomR1 = (int(box[0]+box[2]), int(box[1]+20)) 
        botomR2 = getBotomright(labels)

        # cv2.rectangle(img, topL1, botomR1, (0,255,255), 2)
        # cv2.rectangle(img, topL2, botomR2, (0,255,255), 2)
    else:
        topL   = getTopleft(labels)
        botomR = getBotomright(labels)    

        # cv2.rectangle(img, topL, botomR, (0,255,255), 2)
    _img = copy.deepcopy(img)
    colors = [(0,0,255), (0,255,0), (255,0,0)]
    for label in labels:
        category_id, bbox, segmentation = label
        x, y, w, h = map(int, bbox)
        cv2.rectangle(_img, (x, y), (x + w, y + h), colors[category_id-1], 2)
        cv2.putText(_img, str(category_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[category_id-1], 1)
        saveName = savePath + '/' + f'img{imgId}.jpg'
        cv2.imwrite(saveName, _img)

    # Display the image (press any key to continue)
    # cv2.imshow("Image", _img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # make pins only images
    pinImg = makePinImg(labels, img)

    if 3 in categories:
        pinImg1 = pinImg[topL1[1]:botomR1[1], topL1[0]:botomR1[0]]
        cv2.imshow('Pin 1',pinImg1)
        cv2.waitKey()
        cv2.destroyAllWindows()
        saveName = savePath + '/' + 'Pin1.jpg'
        cv2.imwrite(saveName, pinImg1)
        pinImg2 = pinImg[topL2[1]:botomR2[1], topL2[0]:botomR2[0]]
        cv2.imshow('Pin 2',pinImg2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        saveName = savePath + '/' + 'Pin2.jpg'
        cv2.imwrite(saveName, pinImg2)
    else:
        pinImg1 = pinImg[topL[1]:botomR[1], topL[0]:botomR[0]]
        cv2.imshow('Pin 1',pinImg1)
        cv2.waitKey()
        cv2.destroyAllWindows()
        saveName = savePath + '/' + 'Pin11.jpg'
        cv2.imwrite(saveName, pinImg1)    
    
    # make burr region only images
    burImg = makeBurImg(labels, img)
    if 3 in categories:
        burImg1 = burImg[topL1[1]:botomR1[1], topL1[0]:botomR1[0]]
        cv2.imshow('Bur',burImg1)
        cv2.waitKey()
        cv2.destroyAllWindows()
        saveName = savePath + '/' + 'burr.jpg'
        cv2.imwrite(saveName, burImg1)
    else:    
        burImg1 = burImg[topL[1]:botomR[1], topL[0]:botomR[0]]
        cv2.imshow('Bur', burImg1)
        cv2.waitKey()
        cv2.destroyAllWindows()
        saveName = savePath + '/' + 'burr11.jpg'
        cv2.imwrite(saveName, burImg1)    



    print()        