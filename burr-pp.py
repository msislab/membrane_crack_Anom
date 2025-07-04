import os
import cv2
import glob
import numpy as np
import logging
import matplotlib.pyplot as plt

def calculate_burr_scanline_thickness(image, predictions):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    horizontal_thicknesses = []
    vertical_thicknesses = []
    white_dark_ratios = []
    ratios = []

    img_height, img_width = image.shape[:2]

    for idx, pred in enumerate(predictions):
        poly = np.array([[pred[0], pred[1]],
                        [pred[2], pred[3]],
                        [pred[4], pred[5]],
                        [pred[6], pred[7]]], dtype=np.int32)

        x, y, w, h = cv2.boundingRect(poly)

        # crop_1 = image[y:y+h, x:x+w]

        cx, cy = x + w // 2, y + h // 2
        reduced_w = max(int(w * 0.9), 1)
        reduced_h = max(int(h * 0.9), 1)

        x_start = max(cx - reduced_w // 2, 0)
        y_start = max(cy - reduced_h // 2, 0)
        x_end = min(cx + reduced_w // 2, img_width)
        y_end = min(cy + reduced_h // 2, img_height)

        # crop_2 = image[y_start:y_end, x_start:x_end]

        poly_shifted = poly - [x_start, y_start]

        crop_w = x_end - x_start
        crop_h = y_end - y_start
        mask_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillPoly(mask_crop, [poly_shifted], 255)
        
        # Debug prints
        print("Mask shape:", mask_crop.shape)
        print("Unique values in mask:", np.unique(mask_crop))
        print("Min value:", np.min(mask_crop))
        print("Max value:", np.max(mask_crop))
        print("Poly shifted shape:", poly_shifted.shape)
        print("Poly shifted values:", poly_shifted)

        gray_crop = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
        masked_crop = cv2.bitwise_and(gray_crop, gray_crop, mask=mask_crop)
        _, binary_crop = cv2.threshold(masked_crop, 25, 255, cv2.THRESH_BINARY)

        # Plot the three crops side by side
        plt.figure(figsize=(25, 5))
        
        plt.subplot(141)
        plt.imshow(mask_crop, cmap='gray', vmin=0, vmax=255)
        plt.title('Mask')
        # plt.colorbar()  # Add colorbar to verify value range

        plt.subplot(142)
        plt.imshow(gray_crop, cmap='gray')
        plt.title('Gray Crop')

        plt.subplot(143)
        plt.imshow(masked_crop, cmap='gray')
        plt.title('Masked Crop')

        plt.subplot(144)
        plt.imshow(binary_crop, cmap='gray')
        plt.title('Binary Crop')
        
        
        plt.tight_layout()
        plt.show()
        
        # h_thick, v_thick = self.calculate_scanline_thickness(binary_crop)
        w, l = calculate_scanline_thickness(binary_crop)
        # horizontal_thicknesses.append(h_thick)
        # vertical_thicknesses.append(v_thick)
        horizontal_thicknesses.append(w)
        vertical_thicknesses.append(l)
        if l>0:
            # ratios.append(min(h_thick,v_thick)/max(v_thick,h_thick))
            if w < 5 and l < 5:
                if w == l:
                    ratios.append(0)
                elif w > l:
                    ratios.append(l/w)
            else:
                ratios.append(w/l)
        else:
            ratios.append(0)

        # White/Dark pixel ratio
        binary_crop = cv2.morphologyEx(binary_crop, cv2.MORPH_ERODE, kernel, iterations=1)
        white_pixels = np.count_nonzero(binary_crop == 255)
        total_pixels = binary_crop.size
        dark_pixels = total_pixels - white_pixels

        if white_pixels > 0:
            ratio = dark_pixels / white_pixels
        else:
            ratio = float('inf')

        white_dark_ratios.append(ratio)
        # print('yes')

    return horizontal_thicknesses, vertical_thicknesses, ratios, white_dark_ratios

def calculate_scanline_thickness(mask):
    """
    Calculates the max thickness of white region in binary mask along:
    - Horizontal (row-wise)
    - Vertical (column-wise)
    """
    horizontal_thickness = 0
    vertical_thickness   = 0

    # Find extremes of mask coordinates (length of the burr)
    try:
        white_pixels = np.where(mask > 0)
        # if len(white_pixels[0]) > 0:  # Check if any white pixels exist
        top_y    = np.min(white_pixels[0])
        bottom_y = np.max(white_pixels[0])
        _l1      = bottom_y - top_y
        left_x   = np.min(white_pixels[1])
        right_x  = np.max(white_pixels[1])
        _l2      = right_x - left_x

        l = max(_l1, _l2)
    except Exception as e:
        print(f"Error in calculate_scanline_thickness: {e}")
        l = 0

    h, w = mask.shape

    # Horizontal scan: row by row
    for y in range(h):
        row = mask[y, :]
        in_white = False
        length = 0
        max_length = 0

        for pixel in row:
            if pixel > 0:  # white pixel
                length += 1
                in_white = True
            else:
                if in_white:
                    max_length = max(max_length, length)
                    length = 0
                    in_white = False
        # End of row
        if in_white:
            max_length = max(max_length, length)
        horizontal_thickness = max(horizontal_thickness, max_length)

    # Vertical scan: column by column
    for x in range(w):
        col = mask[:, x]
        in_white = False
        length = 0
        max_length = 0

        for pixel in col:
            if pixel > 0:
                length += 1
                in_white = True
            else:
                if in_white:
                    max_length = max(max_length, length)
                    length = 0
                    in_white = False
        if in_white:
            max_length = max(max_length, length)
        vertical_thickness = max(vertical_thickness, max_length)
    w = min(horizontal_thickness, vertical_thickness)
    # return horizontal_thickness, vertical_thickness
    return w, l

def main(path):
    files = glob.glob(os.path.join(path, "*.jpg"))
    for file in files:
        labelpath = file.replace("jpg", "txt")
        img = cv2.imread(file)
        height, width, _ = img.shape

        dummy_preds = []
        
        with open(labelpath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line)
                x1, y1, x2, y2, x3, y3, x4, y4 = int(x1*width), int(y1*height), int(x2*width), int(y2*height), int(x3*width), int(y3*height), int(x4*width), int(y4*height)
                # Convert points to numpy array format for cv2.polylines
                # pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32)
                # pts = pts.reshape((-1,1,2))
                
                # # Draw the oriented bounding box
                # cv2.polylines(img, [pts], True, (0,255,0), 1)
                dummy_preds.append([x1, y1, x2, y2, x3, y3, x4, y4])
        dummy_preds = np.vstack(dummy_preds)

        horizontal_thicknesses, vertical_thicknesses, ratios, white_dark_ratios = calculate_burr_scanline_thickness(img, dummy_preds)

        print(horizontal_thicknesses)
        print(vertical_thicknesses)
        print(ratios)
        print(white_dark_ratios)



        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print()


if __name__ == "__main__":
    path = '/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/post-processing-test-data'
    main(path)