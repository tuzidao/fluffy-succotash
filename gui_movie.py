import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from ultralytics import YOLO
import threading
import queue
import time

# ===================== 加载你的模型 =====================
MODEL_PATH = r"C:\Users\Administrator\Desktop\PythonProject\finally_model\epoch70.pt"  # 自动找你训练好的模型
MODEL_PATH = r'C:\Users\Administrator\Desktop\PythonProject\finally_model\runs\detect\train\weights\best.pt'
model = YOLO(MODEL_PATH)

# 视频队列（防止GUI卡顿）
frame_queue = queue.Queue(maxsize=2)
stop_flag = False

# ===================== 主界面 =====================
root = tk.Tk()
root.title("驾驶员行为检测系统 v1.0")
root.geometry("900x700")
root.configure(bg="#f0f0f0")

# 状态标签
status_label = ttk.Label(root, text="状态：等待选择视频", font=("微软雅黑", 12))
status_label.pack(pady=10)

# 画布显示视频
canvas = tk.Canvas(root, width=800, height=500, bg="black")
canvas.pack()

# ===================== 按钮区域 =====================
frame_btns = ttk.Frame(root)
frame_btns.pack(pady=15)

def select_video():
    path = filedialog.askopenfilename(
        filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
    )
    if path:
        video_path.set(path)
        status_label.config(text=f"已选择：{path}")

def start_detection():
    global stop_flag
    stop_flag = False
    path = video_path.get()
    if not path:
        status_label.config(text="请先选择视频！")
        return
    status_label.config(text="正在检测中...")
    threading.Thread(target=run_detection, args=(path,), daemon=True).start()

def stop_detection():
    global stop_flag
    stop_flag = True
    status_label.config(text="已停止")

video_path = tk.StringVar()

btn_select = ttk.Button(frame_btns, text="选择视频", command=select_video)
btn_select.grid(row=0, column=0, padx=10)

btn_start = ttk.Button(frame_btns, text="开始检测", command=start_detection)
btn_start.grid(row=0, column=1, padx=10)

btn_stop = ttk.Button(frame_btns, text="停止检测", command=stop_detection)
btn_stop.grid(row=0, column=2, padx=10)

# ===================== 检测主逻辑 =====================
def run_detection(video_path_str):
    global stop_flag
    cap = cv2.VideoCapture(video_path_str)

    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 检测
        results = model(frame, imgsz=320)
        frame_det = results[0].plot()

        # 转RGB
        frame_rgb = cv2.cvtColor(frame_det, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # 缩放到画布
        scale = min(800/w, 500/h)
        new_w, new_h = int(w*scale), int(h*scale)
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

        # 转PhotoImage
        from PIL import Image, ImageTk
        img = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)

        # 刷新画布
        canvas.delete("all")
        canvas.create_image(400, 250, image=imgtk)
        canvas.imgtk = imgtk

        time.sleep(0.03)

    cap.release()
    status_label.config(text="检测完成")

# ===================== 运行 =====================
if __name__ == "__main__":
    root.mainloop()