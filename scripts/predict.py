from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("runs/detect/train12/weights/best.pt")

# Inference on a test image
image_path = "data/basketball-player-detection.v4i.yolov8/test/images/suggested-OkHELcZMcDjRYc8Cx3U9_jpg.rf.4c6600d5d7ef15b4b4a13a2917f3dda2.jpg"
image = cv2.imread(image_path)

# Run inference
results = model.predict(source=image, save=True, conf=0.25)

# Plot Predcitions on the image
annotated_img = results[0].plot()
print("Prediction saved in: ", results[0].save_dir)
