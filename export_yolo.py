from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/zafar/membrane_crack_Anom/m07_pinROI.pt")

# Export the model to TensorRT format
model.export(format="engine",
             imgsz=640,
             batch=32,
             data="/home/zafar/membrane_crack_Anom/roiData.yaml")  # creates 'yolo11n.engine'