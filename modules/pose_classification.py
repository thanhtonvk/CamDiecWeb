from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes):
        super(MLP, self).__init__()

        # Thêm nhiều lớp fully connected (Linear)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, num_classes)

        # Sử dụng hàm kích hoạt ReLU cho các lớp ẩn
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # Forward pass qua các lớp
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)  # Lớp cuối cùng không có ReLU (cho đầu ra)
        out = self.softmax(out)
        return out
def resize(pose):
    new_arr = np.zeros(204)
    new_arr[:len(pose)] = pose
    return new_arr
labels = ['thich',
 'binh_thuong',
 'hieu',
 'chi_gai',
 'nho',
 'nha',
 'yeu',
 'xin_chao',
 'xin_loi',
 'khong_thich',
 'so',
 'biet',
 'me',
 'hen_gap_lai',
 'khoe',
 'cam_on',
 'anh_trai',
 'tam_biet',
 'to_mo']
input_size = 204 
hidden_size = 64  # lớp ẩn với 64 neuron
num_classes = 19
model_cls = MLP(input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64, num_classes=num_classes)
model_cls.load_state_dict(torch.load('models/model_mlp.pth'))
model_cls.eval()
model_pose = YOLO('yolo11n-pose.pt')
def predict_pose(image, conf = 0.9):
    result = model_pose.predict(image,verbose = False)
    pose = result[0].keypoints.xyn.cpu().detach().numpy().flatten().astype('float')
    if len(pose)>0:
        pose_tensor = torch.tensor(resize(pose)).unsqueeze(0).float()
        with torch.no_grad():
            outputs = model_cls(pose_tensor)
            prob, predicted_class = outputs.max(1)
            idx = predicted_class.item()
            prob = prob.item()
            name = labels[idx]
        if prob>conf:
            return name, int(prob*100)
    return 'binh thuong', 100