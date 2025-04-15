import os
import cv2

def convert_webp_to_png(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.webp'):
                webp_path = os.path.join(subdir, file)
                png_path = os.path.join(subdir, os.path.splitext(file)[0] + '.png')
                
                try:
                    # Read the .webp image
                    img = cv2.imread(webp_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        raise ValueError("Failed to read the image file.")
                    
                    # Save as .png
                    cv2.imwrite(png_path, img)
                    print(f"Converted: {webp_path} -> {png_path}")
                except Exception as e:
                    print(f"Failed to convert {webp_path}: {e}")

if __name__ == "__main__":
    root_directory = "/home/zafar/old_pc/data_sets/robot-project-datasets/code-integration/AIRobot/20250329_LineA"
    convert_webp_to_png(root_directory)