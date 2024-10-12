import os
from datetime import datetime
import torch
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv
import pygame
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from time import time
from modules.FaceDetector import FaceDetector
from modules.emotion_recognition import EmotionRecognition
faceDetector = FaceDetector()
emotionRecognition = EmotionRecognition()
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Màu xanh lá cây
thickness = 2
def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

class ObjectDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.DEFAULT, thickness=3)
        self.create_output_dir()

    def create_output_dir(self):
        # Tạo thư mục lưu video và ảnh với tên duy nhất dựa trên thời gian hiện tại
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join('video', current_time)
        os.makedirs(self.output_dir, exist_ok=True)
        self.video_writer = None

    def load_model(self):
        model = YOLO("models/yolov8stonghop10.pt")  # tải mô hình YOLOv8n đã được huấn luyện
        model.to(self.device)
        return model

    def predict(self, frame):
        results = self.model(frame, conf=0.85)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = np.empty((0, 4), dtype=float)
        confidences = np.empty((0,), dtype=float)
        class_ids = np.empty((0,), dtype=int)

        # Trích xuất phát hiện cho từng kết quả
        for result in results:
            boxes = result.boxes.cpu().numpy()
            if len(boxes) == 0:
                continue
            if len(result.boxes.cls) > 0:
                class_id = int(result.boxes.cls[0].item())
            else:
                continue  # bỏ qua nếu không phát hiện đối tượng nào

            xyxys = np.concatenate((xyxys, result.boxes.xyxy.cpu().numpy()), axis=0)
            confidences = np.concatenate((confidences, result.boxes.conf.cpu().numpy()), axis=0)
            class_ids = np.concatenate((class_ids, result.boxes.cls.cpu().numpy().astype(int)), axis=0)

        # Thiết lập phát hiện cho hình ảnh hóa
        detections = sv.Detections(
            xyxy=xyxys,
            confidence=confidences,
            class_id=class_ids,
        )
        # Định dạng nhãn tùy chỉnh
        self.labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence * 100:.2f}%"
            for _, confidence, class_id in zip(xyxys, confidences, class_ids)
        ]

        # Chú thích và hiển thị khung hình
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame

    def __call__(self, frame):
        results = self.predict(frame)
        frame = self.plot_bboxes(results, frame)
        self.save_frame(frame)
        return frame

    def save_frame(self, frame):
        if self.video_writer is None:
            height, width, _ = frame.shape
            video_path = os.path.join(self.output_dir, 'output_video.avi')
            self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
        self.video_writer.write(frame)

    def release_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

class Camera:
    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)  # Sử dụng 0 cho camera mặc định hoặc đường dẫn tệp video

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0
        self.camera = None
        self.detector = ObjectDetection()

        self.canvas_width = 800
        self.canvas_height = 600

        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height, bg="green", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.red_border = self.canvas.create_rectangle(5, 5, self.canvas_width - 5, self.canvas_height - 5, outline="red", width=5)

        button_frame = tk.Frame(window, bg="white")
        button_frame.pack(pady=10)

        button_style = {"bg": "yellow", "fg": "darkblue", "font": ("Helvetica", 16, "bold"), "height": 1, "anchor": "center"}

        self.btn_select_video = tk.Button(button_frame, text="Video", width=8, **button_style, command=self.load_video)
        self.btn_select_video.grid(row=0, column=0, padx=3)

        self.btn_camera = tk.Button(button_frame, text="Camera", width=8, **button_style, command=self.start_camera)
        self.btn_camera.grid(row=0, column=1, padx=3)

        self.btn_zoom_in = tk.Button(button_frame, text="Phóng to", width=8, **button_style, command=self.zoom_in)
        self.btn_zoom_in.grid(row=0, column=2, padx=3)

        self.btn_zoom_out = tk.Button(button_frame, text="Thu nhỏ", width=8, **button_style, command=self.zoom_out)
        self.btn_zoom_out.grid(row=0, column=3, padx=3)

        self.btn_exit = tk.Button(button_frame, text="Thoát", width=8, **button_style, command=self.quit)
        self.btn_exit.grid(row=0, column=4, padx=3)

        self.prev_time = time()
        self.update()
        self.window.mainloop()

    def load_video(self):
        video_path = filedialog.askopenfilename()
        if video_path:
            self.video_source = video_path
            self.camera = Camera(self.video_source)
            self.detector.create_output_dir()  # Tạo thư mục mới khi tải video

    def start_camera(self):
        self.video_source = 0
        self.camera = Camera(self.video_source)
        self.detector.create_output_dir()  # Tạo thư mục mới khi bắt đầu camera

    def zoom_in(self):
        self.canvas_width = int(self.canvas_width * 1.2)
        self.canvas_height = int(self.canvas_height * 1.2)
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)
        self.update_border()

    def zoom_out(self):
        self.canvas_width = int(self.canvas_width / 1.2)
        self.canvas_height = int(self.canvas_height / 1.2)
        self.canvas.config(width=self.canvas_width, height=self.canvas_height)
        self.update_border()

    def update_border(self):
        self.canvas.coords(self.red_border, 5, 5, self.canvas_width - 5, self.canvas_height - 5)

    def quit(self):
        if self.camera:
            self.camera.__del__()
        self.detector.release_video_writer()
        self.window.destroy()

    def update(self):
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                frame = self.detector(frame)
                frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                boxes = faceDetector.detect(frame)
                if len(boxes) > 0:
                    for box in boxes:
                        x_min, y_min, x_max, y_max = box
                        image = frame[y_min:y_max, x_min:x_max]
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        label, score = emotionRecognition.predict(image)
                        org = (x_min, y_min - 10)
                        text = f'{label} : {score}'
                        cv2.putText(frame, text, org, font,
                                    font_scale, color, thickness)
                        cv2.rectangle(frame, (x_min, y_min),
                                    (x_max, y_max), (0, 255, 0), 2)

                # Chú thích và hiển thị khung hình
                frame = self.box_annotator.annotate(
                    scene=frame, detections=detections, labels=self.labels)

                # Tính toán FPS
                current_time = time()
                fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time

                # Hiển thị FPS trên khung hình
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

if __name__ == "__main__":
    App(tk.Tk(), "Object Detection with YOLO")
