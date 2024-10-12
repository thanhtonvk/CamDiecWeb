import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pygame

# Sử dụng token của bạn từ Hugging Face
HUGGINGFACE_API_TOKEN = "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"

# Cấu hình API
model_name = "gpt2"

# Tải mô hình và tokenizer từ Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained(model_name, token=HUGGINGFACE_API_TOKEN)
gpt_model = GPT2LMHeadModel.from_pretrained(model_name, token=HUGGINGFACE_API_TOKEN)

# Khởi tạo pygame mixer
pygame.mixer.init()

# Tải âm thanh
sound_path_xin_chao = 'amthanh/xin_chao.mp3'
sound_path_tam_biet = 'amthanh/tam_biet.mp3'
sound_path_xin_loi = 'amthanh/rat_vui.mp3'
sound_xin_chao = pygame.mixer.Sound(sound_path_xin_chao)
sound_tam_biet = pygame.mixer.Sound(sound_path_tam_biet)
sound_xin_loi = pygame.mixer.Sound(sound_path_xin_loi)

class ObjectDetection:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("models/yolov8nvietnam06.pt")
        model.to(self.device)
        return model

    def predict(self, frame):
        results = self.model(frame, conf=0.8)
        return results

    def get_class_names(self, results):
        class_names = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for class_id in class_ids:
                if class_id in self.CLASS_NAMES_DICT:
                    label = self.CLASS_NAMES_DICT[class_id]
                    class_names.append(label)
        return class_names

class Camera:
    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame

def play_sound(sound):
    sound.play()

def main():
    cam = Camera(0)
    detector = ObjectDetection()

    detected_words = []
    start_time = None

    prev_time = time()
    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        results = detector.predict(frame)
        class_names = detector.get_class_names(results)

        if class_names and start_time is None:
            start_time = time()

        detected_words.extend(class_names)

        # Hiển thị khung hình từ camera và lớp nhận diện
        for i, class_name in enumerate(class_names):
            cv2.putText(frame, class_name, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Tính toán FPS
        curr_time = time()
        fps = round(1 / (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps}", (10, 30 + 30 * (len(class_names) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Thay đổi kích thước khung hình để tăng kích thước hiển thị và giảm FPS
        resized_frame = cv2.resize(frame, (800, 600))
        cv2.imshow("Camera", resized_frame)

        # Kiểm tra xem đã đủ 3 giây kể từ khi xuất hiện lớp đầu tiên chưa
        if start_time is not None and time() - start_time > 3:
            # Kiểm tra thứ tự xuất hiện của các từ
            if ('xin chao' in detected_words and 
                'tam biet' in detected_words and 
                'cam on' in detected_words):
                
                xin_chao_index = detected_words.index('xin chao')
                tam_biet_index = detected_words.index('tam biet')
                cam_on_index = detected_words.index('cam on')
                
                # Phát âm thanh nếu "xin chao" xuất hiện trước "tam biet" và "tam biet" xuất hiện trước "cam on"
                if xin_chao_index < tam_biet_index < cam_on_index:
                    play_sound(sound_xin_chao)
                
                # Phát âm thanh nếu "tam biet" xuất hiện trước "cam on" và "cam on" xuất hiện trước "xin chao"
                if tam_biet_index < cam_on_index < xin_chao_index:
                    play_sound(sound_tam_biet)

            # Kiểm tra thứ tự xuất hiện của các từ "rat vui duoc gap ban", "khong thich", "hen gap lai"
            if ('rat vui duoc gap ban' in detected_words and 
                'khong thich' in detected_words and 
                'hen gap lai' in detected_words):
                
                vui_gap_ban_index = detected_words.index('rat vui duoc gap ban')
                khong_thich_index = detected_words.index('khong thich')
                hen_gap_lai_index = detected_words.index('hen gap lai')
                
                # Phát âm thanh nếu "rat vui duoc gap ban" xuất hiện trước "khong thich" và "khong thich" xuất hiện trước "hen gap lai"
                if vui_gap_ban_index < khong_thich_index < hen_gap_lai_index:
                    play_sound(sound_xin_loi)

            # Đặt lại danh sách từ đã nhận diện và thời gian bắt đầu
            detected_words.clear()
            start_time = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.__del__()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
