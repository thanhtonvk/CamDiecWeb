import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2
device = "mps" if torch.backends.mps.is_available(
) else "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
labels = ['anh_trai', 'biet', 'binh_thuong', 'cam_on', 'chi_gai', 'hen_gap_lai', 'hieu', 'khoe', 'khong_thich', 'me', 'nha', 'nho', 'so', 'tam_biet', 'thich', 'to_mo', 'xin_chao', 'xin_loi', 'yeu']


def initModel():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, len(labels)),
        nn.Softmax(-1)
    )
    model.load_state_dict(torch.load(
        'models/r18_fine.pt', map_location=device))
    model.to(device)
    model.eval()
    return model


class DeafRecogntion():
    def __init__(self):
        self.model = initModel()

    def predict(self, person: np.ndarray):
        person = cv2.cvtColor(person,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(person).convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            print(outputs)
        prob, predicted_class = outputs.max(1)
        idx = predicted_class.item()
        prob = prob.item()
        name = labels[idx]
        print(name)
        return name, int(prob*100)
