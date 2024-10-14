from ultralytics_v10 import YOLOv10
model = YOLOv10('models/best_v10.pt')
labels = model.names
def predictDeaf(image):
    result = model(image,verbose = False,conf=0.4)[0]
    cls = result.boxes.cls.cpu().detach().numpy()
    boxes = result.boxes.xyxy.cpu().detach().numpy().astype('int')
    if len(boxes)>0:
        return labels.get(cls[0]), boxes[0]
    return 'binh thuong',None
