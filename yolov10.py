import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import pygame
from modules.FaceDetector import FaceDetector
from modules.emotion_recognition import EmotionRecognition
from modules.deaf_yolov10 import predictDeaf
from PIL import Image

faceDetector = FaceDetector()
# deafRecogntion = DeafRecogntion()
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
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def plot_bboxes(self, frame):
        faceBoxes = faceDetector.detect(frame)
        emotion_result = []
        if len(faceBoxes) > 0:
            for box in faceBoxes:
                x_min, y_min, x_max, y_max = box
                image = frame[y_min:y_max, x_min:x_max]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label, score = emotionRecognition.predict(image)
                if score > 0.9:
                    emotion_result.append(label)
                else:
                    emotion_result.append('tu nhien')

        name, box = predictDeaf(frame)
        if box is not None:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)
            # Hiển thị cả class 'tam biet'
            text = f'{name}'
            cv2.putText(frame, text, (x_min, y_min-10), font,
                        font_scale, color, thickness, cv2.LINE_AA)
            if True:
                name = name.replace('_',' ')
                if name == 'so' and name in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'rat vui duoc gap ban' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'khong thich' and 'tuc gian' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'cam on' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'khoe' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'thich' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'xin loi' and 'buon' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'hen gap lai' and 'tu nhien' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'xin chao' and 'tu nhien' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'tam biet' and 'buon' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'nho' and 'buon' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'hieu' and 'tu nhien' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'to mo' and 'bat ngo' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'chi gai' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'anh trai' and 'tu nhien' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'me' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'nha' and 'tu nhien' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'yeu' and 'vui ve' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)
                if name == 'biet' and 'tu nhien' in emotion_result:
                    sound_file = f'amthanh/{name}.mp3'
                    play_sound(sound_file)

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
        frame  = self.plot_bboxes(frame)
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
