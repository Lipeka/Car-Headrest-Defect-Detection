!pip install roboflow ultralytics --upgrade
from roboflow import Roboflow
rf = Roboflow(api_key="CpWubnjT7XDbJcyiIwCR")
project = rf.workspace("headrest-defect").project("headrest-defects")
version = project.version(2)
dataset = version.download("yolov12")
!yolo task=detect mode=train model=yolov8m.pt data=/content/Headrest-Defects-2/data.yaml epochs=50 imgsz=640 batch=8 patience=20 cos_lr=True save_period=5
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
while True:
    uploaded = files.upload()
    if not uploaded:
        print("🚫 No file uploaded. Stopping.")
        break
    file_path = next(iter(uploaded))
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n📂 Uploaded: {file_path}")
    print("🔍 Running detection...")
    !yolo task=detect mode=predict model=runs/detect/train2/weights/best.pt \
        conf=0.10 iou=0.7 source="{file_path}" save=True
    pred_path, pred_dir = get_latest_prediction_file()
    if pred_path is None:
        print("❌ No predictions found. Skipping.")
        continue
    ext_out = os.path.splitext(pred_path)[1].lower()
    if ext_out in [".jpg", ".png"]:
        pred_dest = os.path.join("predictions", f"{base_name}_pred.jpg")
        shutil.move(pred_path, pred_dest)
        print(f"✅ Image prediction saved: {pred_dest}")
        display(Image(filename=pred_dest))
    elif ext_out in [".mp4", ".avi"]:
        pred_dest = os.path.join("predictions", f"{base_name}_pred.mp4")
        shutil.move(pred_path, pred_dest)
        print(f"🎞️ Video prediction saved: {pred_dest}")
        display(Video(filename=pred_dest, embed=True))
    else:
        print(f"⚠️ Unknown output format: {ext_out}")
        continue
    shutil.rmtree(pred_dir, ignore_errors=True)
