import cv2
import glob
import numpy as np

img_path = '/home/zafar/old_pc/data_sets/mvtec_anomaly_detection/carpet/test/color'
mask_path = '/home/zafar/old_pc/data_sets/mvtec_anomaly_detection/carpet/ground_truth/color'

overlay_color = np.array([0, 0, 255], dtype=np.uint8)
for imgPath in glob.glob(f'{img_path}/*.png'):
    name     = imgPath.split('/')[-1]
    maskPath = mask_path + '/' + name.split('.png')[0]+'_mask.png'

    print(imgPath)
    print(maskPath)

    img = cv2.imread(imgPath)
    mask = cv2.imread(maskPath)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_overlay = np.where(mask == 255, overlay_color, img)

    # imgRed = img[:,:,2]
    # imgRed[mask>0] = 255

    # img[:,:,2] = imgRed

    cv2.imshow('', mask_overlay)
    cv2.waitKey()
    cv2.destroyAllWindows()
