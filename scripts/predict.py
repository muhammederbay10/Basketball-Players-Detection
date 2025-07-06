from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("runs/detect/train12/weights/best.pt")

# Inference on a test image
image_path = "data/basketball-player-detection.v4i.yolov8/test/images/suggested-OkHELcZMcDjRYc8Cx3U9_jpg.rf.4c6600d5d7ef15b4b4a13a2917f3dda2.jpg"
image = cv2.imread(image_path)

# Inference o test video
video_path = "data/basketball-player-detection.v4i.yolov8/test/Videos/training.mp4"

# Run inference
results = model.predict(source=video_path, conf=0.6, iou=0.5, save=True)

# Plot Predcitions on the image
annotated_img = results[0].plot()
print("Prediction saved in: ", results[0].save_dir)
results[0].save_crop("output/")
results[0].show
results[0].save("output/image.jpg")