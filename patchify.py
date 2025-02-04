import numpy as np
import cv2

def get_idxs(img_size, div_factor=4, overlap=5):
    """Generates overlapping patch start and end indices."""
    patch_size = img_size // div_factor
    indices = []
    for i, start in enumerate(range(0, img_size, patch_size)):
        end = min(start + patch_size + overlap, img_size) if i == 0 else min(start + patch_size, img_size)
        indices.append([max(0, start - 2 * overlap), end])
        if img_size - end < 50:  # Adjust last patch
            indices[-1][1] = img_size
            break
    return np.array(indices)

def getPatch(img, pinPoses):
    patch = img[pinPoses[1]:pinPoses[3],pinPoses[0]:pinPoses[2]]
    return patch

def patchify(img, div_factor_x=4, div_factor_y=2, overlap=5, patch_size=(640, 640), surface=None, pinPreds=None):
    """Splits the image into overlapping patches and resize"""
    if surface:
        y, x, _ = img.shape
        if surface=='Front11':
            patches, patch_positions = [], []
            pinPoses = [[0,0,425, y], [425,0,875, y], [875,0,1345,y], [1345,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Front12':
            patches, patch_positions = [], []
            pinPoses = [[0,0,425, y], [425,0,875, y], [875,0,1345,y], [1345,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Front21':
            patches, patch_positions = [], []
            pinPoses = [[0,0,415, y], [415,0,865, y], [865,0,1315,y], [1315,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Front22':
            patches, patch_positions = [], []
            pinPoses = [[0,0,465, y], [465,0,865, y], [865,0,1315,y], [1315,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Top11':
            patches, patch_positions = [], []
            pinPoses = [[0,0,425, y], [425,0,870, y], [870,0,1340,y], [1340,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Top12':
            patches, patch_positions = [], []
            pinPoses = [[0,0,425, y], [425,0,870, y], [870,0,1340,y], [1340,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Top21':
            patches, patch_positions = [], []
            pinPoses = [[0,0,345, y], [345,0,795, y], [795,0,1240,y], [1240,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
        elif surface=='Top22':
            patches, patch_positions = [], []
            pinPoses = [[0,0,345, y], [345,0,790, y], [790,0,1240,y], [1240,0,x,y]]
            for i in range(len(pinPoses)):
                patch = getPatch(img,pinPoses[i])
                # cv2.imshow('', patch)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                patches.append(cv2.resize(patch, (640,640)))
                patch_positions.append(pinPoses[i])
    else:
        h, w, _ = img.shape
        y_idxs, x_idxs = get_idxs(h, div_factor_y, overlap), get_idxs(w, div_factor_x, overlap)
        
        patches, patch_positions = [], []
        
        for y1, y2 in y_idxs:
            for x1, x2 in x_idxs:
                patch = img[y1:y2, x1:x2]
                patch_positions.append([x1, y1, x2, y2])  # Store original positions
                patches.append(cv2.resize(patch, patch_size))  # Resize to fixed size
        
    return {"patches": patches, "patch_positions": np.array(patch_positions), "img_shape": img.shape}

def reconstruct_image(patch_data):
    """Reconstructs the image from patches using original positions."""
    img_shape, patch_positions = patch_data["img_shape"], patch_data["patch_positions"]
    reconstructed = np.zeros(img_shape, dtype=np.uint8)

    for i, patch in enumerate(patch_data["patches"]):
        x1, x2, y1, y2 = patch_positions[i]
        patch_resized = cv2.resize(patch, (x2 - x1, y2 - y1))
        reconstructed[y1:y2, x1:x2] = patch_resized  # Place patch back
    
    return reconstructed

if __name__ == "__main__":

    
    # img = np.zeros((1280, 1920, 3), dtype=np.uint8)
    
    # # Drawing example red rectangles for testing
    # cv2.rectangle(img, (10, 15), (30, 45), (0, 0, 255), -1)
    # cv2.rectangle(img, (50, 120), (300, 425), (0, 0, 255), -1)
    # cv2.rectangle(img, (150, 320), (900, 500), (0, 0, 255), -1)

    # cv2.imshow("Original Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # patch_data = patchify(img)
    # reconstructed_img = reconstruct_image(patch_data)

    # cv2.imshow("Reconstructed Image", reconstructed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    imgPath = '/home/zafar/old_pc/data_sets/robot-project-datasets/normal-pin-data/SystemStart-20250110-151428/selected_data/1.png'
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (1920,1280))
    img = img[120:675, 100:1845]

    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    patch_data = patchify(img)
    reconstructed_img = reconstruct_image(patch_data)

    cv2.imshow("Reconstructed Image", reconstructed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()