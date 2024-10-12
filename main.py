import cv2
from modules.FaceDetector import FaceDetector
from modules.emotion_recognition import EmotionRecognition
# Mở camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)
faceDetector = FaceDetector()
emotionRecognition = EmotionRecognition()
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Màu xanh lá cây
thickness = 2

# Kiểm tra nếu camera mở thành công
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    # Đọc từng frame từ camera
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể nhận frame (stream bị gián đoạn). Thoát...")
        break
    boxes = faceDetector.detect(frame)
    if len(boxes)>0:
        box = boxes[0]
        x_min,y_min,x_max,y_max = box
        image = frame[y_min:y_max,x_min:x_max]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label,score = emotionRecognition.predict(image)
        org = (x_min, y_min - 10) 
        text = f'{label} : {score}'
        cv2.putText(frame, text, org, font, font_scale, color, thickness)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Hiển thị frame với bounding box
    cv2.imshow('Camera với Bounding Box', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
