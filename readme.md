
# 🔧 Headrest Defects Detection with YOLOv8 and Roboflow

This pipeline performs **object detection on headrest defects** in images and video files.  
Using **YOLOv8** (Ultralytics) and **Roboflow**, it lets you:

✅ Download a custom headrest defect dataset directly from Roboflow  
✅ Train a state-of-the-art object detector  
✅ Perform batch inference on new images or video files  
✅ Display and save the results in a convenient directory

---

## 🌟 Features

- 🔹 Integrates smoothly with **Roboflow API**
- 🔹 Performs **training and fine-tuning with YOLOv8**
- 🔹 Allows you to **predict on both images and video**
- 🔹 Saves detections to a `predictions/` directory
- 🔹 Displays results directly in **Colab notebook**

---

## 🛠 Tech Stack

- [Roboflow API](https://github.com/roboflow/roboflow-python)  
- [UltralyticsYOLOv8](https://github.com/ultralytics/ultralytics)  
- [Python, Google Colab, shutil, glob]

---

## 🔹 Installation

```bash
pip install roboflow ultralytics --upgrade
````

---

## 🔹 Prepare Dataset and Train Model

```python
from roboflow import Roboflow

# 1️⃣ Authenticate with API key
rf = Roboflow(api_key="your_api_key_here")

# 2️⃣ Select your project and version
project = rf.workspace("your-workspace").project("your-project")
version = project.version(2)

# 3️⃣ Download the dataset in YOLOv8 format
dataset = version.download("yolov12")

# 4️⃣ Train a custom model with downloaded data
!yolo task=detect mode=train model=yolov8m.pt data=/content/Headrest-Defects-2/data.yaml epochs=50 imgsz=640 batch=8 patience=20 cos_lr=True save_period=5
```

---

## 🔹 Perform Predictions

```python
from google.colab import files
import os
from glob import glob
from IPython.display import Image, Video, display
import shutil

os.makedirs("predictions", exist_ok=True)

def get_latest_prediction_file():
    """Returns the most recent file from the latest YOLOv8 prediction folder."""
    pred_dirs = sorted(glob("runs/detect/predict*"), key=os.path.getmtime, reverse=True)
    for pred_dir in pred_dirs:
        pred_files = glob(os.path.join(pred_dir, "*"))
        if pred_files:
            return pred_files[0], pred_dir
    return None, None

# Loop to upload files and run detections
while True:
    uploaded = files.upload()
    if not uploaded:
        print("🚫 No file uploaded. Stopping.")
        break
    
    file_path = next(iter(uploaded))
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n📁 Uploaded: {file_path}")
    print("🔍 Running detection...")

    !yolo task=detect mode=predict model=runs/detect/train2/weights/best.pt conf=0.10 iou=0.7 source="{file_path}" save=True
    
    pred_file, pred_dir = get_latest_prediction_file()
    if pred_file is None:
        print("❌ No predictions found.")
        continue
    
    ext = os.path.splitext(pred_file)[1].lower()
    if ext in [".jpg", ".png"]:
        pred_dest = os.path.join("predictions", f"{base_name}_pred.jpg")
        shutil.move(pred_file, pred_dest)
        print(f"✅ Image prediction saved: {pred_dest}")
        display(Image(filename=pred_dest))
    elif ext in [".mp4", ".avi"]:
        pred_dest = os.path.join("predictions", f"{base_name}_pred.mp4")
        shutil.move(pred_file, pred_dest)
        print(f"🎞 Video prediction saved: {pred_dest}")
        display(Video(filename=pred_dest, embed=True))
    else:
        print(f"⚠ Unknown output format.")
        continue
    
    shutil.rmtree(pred_dir, ignore_errors=True)
```

---

## 🔹 Notes

* The pipeline runs directly in **Google Colab**.
* The **API key and directory paths** should be updated to match your project.
* The **model weights** will be saved in `runs/detect/train2/weights/`.

---

## 🔹 Possible Applications

✅ Quality control in automotive seats
✅ Large scale defect spotting
✅ Reliable, automated pipeline to aid manual inspectors

---

## 📝 License

This project is licensed under the **MIT license** — feel free to reuse and modify.
