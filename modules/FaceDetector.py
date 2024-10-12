import numpy as np
import cv2
from modules.SCRFD import SCRFD
from skimage import transform


def resizeBox(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min

    side_length = max(width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    new_x_min = x_center - side_length / 2
    new_y_min = y_center - side_length / 2
    new_x_max = x_center + side_length / 2
    new_y_max = y_center + side_length / 2
    return new_x_min, new_y_min, new_x_max, new_y_max


def check_and_crop(image, x_min, y_min, x_max, y_max):
    # Kiểm tra tọa độ có hợp lệ hay không
    height, width, _ = image.shape

    if x_min >= x_max or y_min >= y_max:
        return False

    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        return False
    return True


class FaceDetector:
    def __init__(self, ctx_id=0, det_size=(640, 640)):
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.model = SCRFD(model_file="models/scr_face_detector.onnx")
        self.model.prepare()

    def detect(
            self,
            np_image: np.ndarray,
            confidence_threshold=0.5,
    ):
        bboxes = []
        predictions = self.model.get(
            np_image, threshold=confidence_threshold, input_size=self.det_size)
        if len(predictions) != 0:
            for _, face in enumerate(predictions):
                bbox = face["bbox"]
                bbox = resizeBox(bbox)
                if check_and_crop(np_image, bbox[0], bbox[1], bbox[2], bbox[3]):
                    bbox = list(map(int, bbox))
                    bboxes.append(bbox)
                    break
        return bboxes
