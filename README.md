# 🏀 Basketball Player Detection using YOLOv8

This project aims to detect basketball players in images using the YOLOv8 object detection model by [Ultralytics](https://github.com/ultralytics/ultralytics). It applies deep learning to identify players in various on-court scenarios with high accuracy.

## 📂 Project Structure

```

Basketball Players Detection Project/
│
├── data/
│   └── basketball-player-detection.v2i.yolov8/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── valid/
│           ├── images/
│           └── labels/
│
├── runs/                   # Contains training outputs
├── scripts/
│   ├── train.py            # Training script
│   ├── predict.py          # Inference script
│   └── callbacks.py        # Custom training callbacks (e.g., metrics, logging)
│
├── requirements.txt
└── README.md

````

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

## 🏋️‍♂️ Training

To train the YOLOv8 model on the basketball player dataset:

```bash
python scripts/train.py
```

The best weights will be saved to:

```
runs/detect/train*/weights/best.pt
```

## 🔍 Inference

To run inference on test images:

```bash
python scripts/predict.py
```

This will load your trained model and visualize predictions on test images. Annotated images are saved automatically.

## 🧠 Model

* Base model: `yolov8s.pt` (YOLOv8 small)
* Training epochs: 100 (early stopping at 19)
* Achieved mAP50: **0.82**
* Evaluation dataset: 7 validation images

## 🔁 Callbacks

The `callbacks.py` file contains custom callback functions that can be used to extend or modify training behavior, such as:

* Custom logging
* Metric tracking
* Early stopping adjustments

## 📸 Sample Result

Annotated images are saved in `runs/detect/train*/`.

![Sample Output](https://github.com/user-attachments/assets/d029641f-d3d2-465f-9195-f88ec4479ba9)

## 📄 License

This project is open source under the MIT License.

