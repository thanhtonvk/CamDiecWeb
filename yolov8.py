import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import pygame
from modules.FaceDetector import FaceDetector
from modules.emotion_recognition import EmotionRecognition
from modules.deaf_recognition import DeafRecogntion
from PIL import Image
from modules.pose_classification import predict_pose

faceDetector = FaceDetector()
deafRecogntion = DeafRecogntion()
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


def get_person(result, image):
    result = result[0]
    boxes = result.boxes.xyxy
    cls = result.boxes.cls
    if len(boxes) > 0:
        box = boxes[0].cpu().numpy().astype(int)
        cls = cls[0].cpu().numpy()
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            box = (x1, y1, x2, y2)
            image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (224, 224))
        return image, box
    return image, None


class ObjectDetection:
    def __init__(self,ngonngu = 'vi'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.ngonngu = ngonngu
    def load_model(self):
        # tải mô hình YOLOv8n đã được huấn luyện
        model = YOLO("models/yolov8n.pt")
        model.to(self.device)
        return model

    def predict(self, frame):
        results = self.model(frame, conf=0.4, verbose=False)
        return results

    def plot_bboxes(self, results, frame):
        faceBoxes = faceDetector.detect(frame)
        emotion_result = []
        if len(faceBoxes) > 0:
            for box in faceBoxes:
                x_min, y_min, x_max, y_max = box
                image = frame[y_min:y_max, x_min:x_max]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label, score = emotionRecognition.predict(image)
                emotion_result.append(label)

        image, box = get_person(results, frame)
        if box is not None:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)
            name, prob = predict_pose(image,0.9)

            # Hiển thị cả class 'tam biet'
            text = f'{name}-{prob}'
            cv2.putText(frame, text, (x_min, y_min-10), font,
                        font_scale, color, thickness, cv2.LINE_AA)
            if prob >=100:
                sound_file = None
                if name == 'so' and name in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'rat vui duoc gap ban' and 'vui ve' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'khong thich' and 'tuc gian' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'cam on' and 'vui ve' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'khoe' and 'vui ve' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'thich' and 'vui ve' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'xin loi' and 'buon' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'hen gap lai' and 'tu nhien' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'xin chao' and 'tu nhien' in emotion_result:
                    sound_file = f'{name}.mp3'
                if name == 'tam biet' and 'buồn' in emotion_result:
                    sound_file = f'{name}.mp3'
                if sound_file:
                    play_sound(f'amthanh/{self.ngonngu}/{sound_file}')
        for label, box in zip(emotion_result, faceBoxes):
            x_min, y_min, x_max, y_max = box
            org = (x_min, y_min - 10)
            text = f'{label}'
            cv2.putText(frame, text, org, font,
                        font_scale, color, thickness)
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
        return frame

    def __call__(self, frame):
        results = self.predict(frame)
        frame = self.plot_bboxes(results, frame)
        return frame
class Camera:
    def __init__(self, video_source=0):
        # Sử dụng 0 cho camera mặc định hoặc đường dẫn tệp video
        self.video = cv2.VideoCapture(video_source)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame
