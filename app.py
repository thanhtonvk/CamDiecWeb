from importlib import import_module
import os
from flask import Flask, render_template, Response, request, redirect, url_for
from yolov8 import Camera, ObjectDetection
import time
import cv2




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

source = 'camera'  # Mặc định là sử dụng camera

@app.route("/")
def index():
    """Trang chủ phát video."""
    return render_template("index.html")

@app.route("/set_source", methods=["POST"])
def set_source():
    global source
    source = request.form['source']
    return '', 204

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        global source
        source = 'uploaded_video'
        app.config['UPLOADED_VIDEO_PATH'] = file_path
        return redirect(url_for('index'))
ngon_ngu = ['vi','en','cn']
def gen(camera):
    """Hàm generator phát video."""
    detector = ObjectDetection(ngonngu='vi')
    frame_count = 0
    start_time = time.time()

    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        frame = detector(frame)

        # Tính toán FPS
        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time

        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0

        # Làm tròn FPS về số nguyên
        fps_int = int(round(fps))

        # Vẽ FPS lên khung hình
        cv2.putText(frame, f"FPS: {fps_int}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mã hóa khung hình ở định dạng JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Đặt lại bộ đếm và thời gian bắt đầu sau mỗi giây
        if elapsed_time > 1:
            frame_count = 0
            start_time = time.time()

@app.route("/video_feed")
def video_feed():
    """Route phát video. Đặt cái này vào thuộc tính src của thẻ img."""
    global source
    if source == 'video':
        video_path = "C:/Users/84986/Desktop/hotrogiaotiepcamdiec/testVD.mp4"  # Đường dẫn tới video của bạn
        camera = Camera()
        camera.video = cv2.VideoCapture(video_path)
    elif source == 'uploaded_video':
        video_path = app.config.get('UPLOADED_VIDEO_PATH')
        camera = Camera()
        camera.video = cv2.VideoCapture(video_path)
    else:
        camera = Camera()
    return Response(gen(camera), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host="0.0.0.0", threaded=True)
