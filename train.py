import torch
from ultralytics import YOLO
import multiprocessing
import os

os.environ["ULTRALYTICS_DOWNLOAD"] = "0"
multiprocessing.freeze_support()

if __name__ == '__main__':

    # ========== 请修改为你的实际路径 ==========
    DATA_YAML = r"D:\深度学习\yolov8\W\model_train\driver_behavior.yaml"

    # 如果 yolov8s.pt 已下载，建议用绝对路径
    # 如果放在 train.py 同级目录，可以直接写 "yolov8s.pt"
    model = YOLO("yolov8s.pt", task="detect")

    model.train(
        data=DATA_YAML,
        epochs=20,
        imgsz=320,
        batch=16,
        device="cpu",               # 如果没 GPU，改成 "cpu"
        workers=2,

        pretrained=True,
        amp=False,
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.05,
        weight_decay=0.0005,

        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        fliplr=0.5,

        patience=10,
        label_smoothing=0.1,
        cache="ram",

        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True
    )