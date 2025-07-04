from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("runs/detect/train12/weights/best.pt")

# Inference on a test image
image_path = "data/basketball-player-detection.v2i.yolov8/test/images/suggested-uvSFq63zT73UH7Q9HC0L_jpg.rf.552147feba9707ac6d6c4525b3bb34c8.jpg"
image = cv2.imread(image_path)

# Run inference
results = model.predict(source=image, save=True, conf=0.25)

# Plot Predcitions on the image
annotated_img = results[0].plot()
print("Prediction saved in: ", results[0].save_dir)
