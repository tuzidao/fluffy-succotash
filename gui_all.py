import tkinter as tk
from tkinter import filedialog
import cv2
import time
from PIL import Image, ImageTk
from model1 import DriverBehaviorModel


class DistractionMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("驾驶员分心行为检测系统")
        self.root.geometry("1100x700")
        self.root.configure(bg="#f0f2f5")

        self.model = DriverBehaviorModel()

        self.cap = None
        self.current_img = None
        self.media_type = None
        self.is_detecting = False

        self.setup_ui()

    def setup_ui(self):
        # 左侧画面
        left_frame = tk.Frame(self.root, bg="white", bd=2, relief=tk.RIDGE)
        left_frame.place(x=20, y=20, width=720, height=560)
        tk.Label(left_frame, text="检测画面", font=("微软雅黑", 14, "bold"), bg="white").pack(pady=5)
        self.canvas = tk.Canvas(left_frame, bg="#e6e6e6", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 状态栏
        status_frame = tk.Frame(self.root, bg="white", bd=2, relief=tk.RIDGE)
        status_frame.place(x=20, y=590, width=720, height=70)
        self.status_lab = tk.Label(status_frame, text="状态：等待检测", font=("微软雅黑", 14, "bold"), fg="black", bg="white")
        self.status_lab.pack(fill=tk.X, pady=15)

        # 右侧面板
        right_panel = tk.Frame(self.root, bg="white", bd=2, relief=tk.RIDGE)
        right_panel.place(x=760, y=20, width=310, height=640)

        # 按钮
        btn_w = 120
        btn_h = 38
        tk.Button(right_panel, text="📷 选择照片", font=("微软雅黑", 10), width=12, height=2,
                  command=self.load_photo).place(x=25, y=30, width=btn_w, height=btn_h)
        tk.Button(right_panel, text="🎬 选择视频", font=("微软雅黑", 10), width=12, height=2,
                  command=self.load_video).place(x=160, y=30, width=btn_w, height=btn_h)

        tk.Button(right_panel, text="📹 打开摄像头", font=("微软雅黑", 10), width=12, height=2,
                  command=self.open_camera).place(x=25, y=80, width=260, height=btn_h)

        tk.Button(right_panel, text="▶ 开始检测", bg="#27ae60", fg="white", font=("微软雅黑", 10),
                  width=12, height=2, command=self.start_detect).place(x=25, y=135, width=btn_w, height=btn_h)
        tk.Button(right_panel, text="⏹ 停止检测", bg="#e74c3c", fg="white", font=("微软雅黑", 10),
                  width=12, height=2, command=self.stop_detect).place(x=160, y=135, width=btn_w, height=btn_h)

        # 结果显示
        result_frame = tk.Frame(right_panel, bg="white", bd=1, relief=tk.RIDGE)
        result_frame.place(x=15, y=195, width=280, height=420)

        tk.Label(result_frame, text="📊 检测结果", font=("微软雅黑", 14, "bold"), bg="white", fg="#2c3e50").pack(pady=12)

        self.behavior_lab = tk.Label(result_frame, text="行为：无", font=("微软雅黑", 18, "bold"), bg="white", fg="black")
        self.behavior_lab.pack(pady=25)

        self.detail_lab = tk.Label(result_frame, text="", font=("微软雅黑", 12), bg="white", fg="gray")
        self.detail_lab.pack(pady=5)

        # 分隔线
        tk.Frame(result_frame, height=2, bg="#e0e0e0").pack(fill=tk.X, pady=15)

        self.info_lab = tk.Label(result_frame, text="支持检测：\n🚬 抽烟  📱 玩手机  🥤 喝水", 
                                  font=("微软雅黑", 11), bg="white", fg="#555", justify=tk.LEFT)
        self.info_lab.pack(pady=10)

    def load_photo(self):
        path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg;*.png;*.jpeg")])
        if not path:
            return
        self.cap = None
        self.media_type = "photo"
        self.current_img = cv2.imread(path)
        self.start_detect()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("视频", "*.mp4;*.avi;*.mov;*.mkv")])
        if not path:
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        self.media_type = "video"
        self.current_img = None
        self.start_detect()

    def open_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.media_type = "camera"
        self.current_img = None
        self.start_detect()

    def start_detect(self):
        if self.is_detecting or self.media_type is None:
            return
        self.is_detecting = True
        self.status_lab.config(text="状态：检测中...", fg="#27ae60")
        if self.media_type == "photo":
            self.process_photo()
        else:
            self.update_stream()

    def stop_detect(self):
        self.is_detecting = False
        self.status_lab.config(text="状态：已停止", fg="black")
        if self.cap:
            self.cap.release()

    def get_class_color(self, cls_name):
        colors = {"face": (0, 255, 0), "phone": (0, 0, 255), 
                  "smoke": (0, 140, 255), "drink": (255, 0, 0)}
        return colors.get(cls_name, (255, 255, 0))

    def draw_boxes(self, img, labels, boxes, confs):
        img_copy = img.copy()
        for label, box, conf in zip(labels, boxes, confs):
            x, y, w, h = box
            color = self.get_class_color(label)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img_copy, f"{label} {conf:.2f}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img_copy

    def process_photo(self):
        img = self.current_img.copy()
        labels, boxes, confs = self.model.detect(img)
        img_draw = self.draw_boxes(img, labels, boxes, confs)

        # 更新界面
        if labels:
            # 过滤掉 face
            behaviors = [l for l in labels if l != "face"]
            if behaviors:
                self.behavior_lab.config(text=f"行为：{', '.join(behaviors)}", fg="red")
                self.detail_lab.config(text=f"检测到 {len(behaviors)} 个行为")
                self.status_lab.config(text="状态：检测完成 ✅", fg="#27ae60")
            else:
                self.behavior_lab.config(text="行为：无（仅检测到人脸）", fg="black")
                self.detail_lab.config(text="")
        else:
            self.behavior_lab.config(text="行为：无", fg="black")
            self.detail_lab.config(text="未检测到任何目标")

        self.show_frame(img_draw)
        self.is_detecting = False

    def update_stream(self):
        if not self.is_detecting or not self.cap:
            return
        ret, img = self.cap.read()
        if not ret:
            self.stop_detect()
            return

        labels, boxes, confs = self.model.detect(img)
        img_draw = self.draw_boxes(img, labels, boxes, confs)

        # 更新行为显示
        if labels:
            behaviors = [l for l in labels if l != "face"]
            if behaviors:
                self.behavior_lab.config(text=f"行为：{', '.join(behaviors)}", fg="red")
                self.detail_lab.config(text=f"检测到 {len(behaviors)} 个行为")
            else:
                self.behavior_lab.config(text="行为：无（仅检测到人脸）", fg="black")
                self.detail_lab.config(text="")
        else:
            self.behavior_lab.config(text="行为：无", fg="black")
            self.detail_lab.config(text="")

        self.show_frame(img_draw)
        self.root.after(50, self.update_stream)

    def show_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        max_w, max_h = 680, 520
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        x = (680 - nw) // 2
        y = (520 - nh) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)


if __name__ == "__main__":
    root = tk.Tk()
    app = DistractionMonitorGUI(root)
    root.mainloop()