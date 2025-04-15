import cv2
import numpy as np

def preProcess_patch(img, surfaceName='', scale_factor=1.0):

    # if 'Front' in surfaceName:
    #     bThres = BThres[surfaceName][f'{patchIdx}'] - 2
    # elif 'Top' in surfaceName:
    #     bThres = BThres[surfaceName][f'{patchIdx}'] - 1.2      
    
    # BGR to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # print(surfaceName)
    # split the image into its channels
    h, s, v = cv2.split(img)

    # brightness = np.sum(v)/(v.shape[0]*v.shape[1])

    # diff = brightness - bThres

    # if diff>0:
        # scale_factor = 1 + (diff/25)
        # if scale_factor > 1.7:
        #     # TODO: give warning of too much higher brightness
        #     scale_factor = 1.7
        #     v = np.clip(v * 0.9, 0, 255).astype(np.uint8)
    
    # gamma correction
    v = np.array(255*(v / 255) ** scale_factor, dtype = 'uint8')
    
    # Hue and Saturation adjustment
    h = np.mod(h * 1.15, 180).astype(np.uint8)
    s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
    
    # merge the channels back and convert to BGR
    img = cv2.merge([h,s,v])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # scale one channel more in BGR to enhance the color more
    # img[:,:,2] = np.clip(img[:,:,2]*1.25, 0, 255).astype(np.uint8)

    # Test this when needed
    # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img