import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set the directory containing images
image_dir = "/home/zafar/old_pc/data_sets/robot-project-datasets/pin_anomaly_data/new_data_factory_bldng/lineA/nightShift-03-06/AbrasionAnomaly/anomaly_only"  # Change this to your directory path
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# List to store color data for all images
bgr_colors = []

top_left = None
selecting = False

def select_roi(event, x, y, flags, param):
    global image, roi_colors, top_left, selecting

    if event == cv2.EVENT_LBUTTONDOWN:  # First click - Start selection
        top_left = (x, y)
        selecting = True

    elif event == cv2.EVENT_LBUTTONUP and selecting:  # Second click - End selection
        selecting = False
        bottom_right = (x, y)

        x1, y1 = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
        x2, y2 = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])

        # Extract the selected patch
        patch = image[y1:y2, x1:x2]

        if patch.size == 0:
            print("Invalid ROI selected. Try again.")
            return

        # Convert to HSV
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # Compute mean color in BGR and HSV
        mean_bgr = np.mean(patch, axis=(0, 1))
        # mean_hsv = np.mean(hsv_patch, axis=(0, 1))

        # Store the colors
        bgr_colors.append(mean_bgr)

        # Draw a rectangle on selected ROI
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show updated image
        cv2.imshow("Image", image)

# Process each image in the directory
for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (1920,1280))
    
    if image is None:
        print(f"Skipping {img_file}, unable to load.")
        continue

    clone = image.copy()
    roi_colors = []  # Reset for each image

    # Show the image and set mouse callback
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", select_roi)

    # Wait for 'q' to move to next image
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # # Store colors for this image
    # if roi_colors:
    #     all_roi_colors[img_file] = roi_colors

cv2.destroyAllWindows()

bgr_colors = np.array(bgr_colors)
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(bgr_colors[:, 0], bgr_colors[:, 1], bgr_colors[:, 2], 
            c=bgr_colors / 255.0, marker='o', s=50)
ax1.set_xlabel("Blue")
ax1.set_ylabel("Green")
ax1.set_zlabel("Red")
ax1.set_title("BGR Color Space")
# Plot the collected colors
# if all_roi_colors:
    # fig_height = max(6, len(all_roi_colors) * 3)  # Adjust height dynamically
    # plt.figure(figsize=(12, fig_height))
    
    # for idx, (img_name, roi_colors) in enumerate(all_roi_colors.items()):
    #     bgr_colors, hsv_colors = zip(*roi_colors)

    #     # Convert BGR to RGB for visualization
    #     rgb_colors = [color[::-1] / 255.0 for color in bgr_colors]

    #     plt.subplot(len(all_roi_colors), 2, idx * 2 + 1)
    #     plt.title(f"Selected Colors ")
    #     plt.imshow([rgb_colors])
    #     plt.axis("off")

    #     # Plot HSV Values
    #     # plt.subplot(len(all_roi_colors), 2, idx * 2 + 2)
    #     # plt.title(f"Mean HSV Values")
    #     # plt.plot(range(len(hsv_colors)), [h[0] for h in hsv_colors], 'r-', label='Hue')
    #     # plt.plot(range(len(hsv_colors)), [h[1] for h in hsv_colors], 'g-', label='Saturation')
    #     # plt.plot(range(len(hsv_colors)), [h[2] for h in hsv_colors], 'b-', label='Value')
    #     # plt.legend()

    # plt.tight_layout()
plt.show()
print()