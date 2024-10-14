
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image
from unidecode import unidecode
device = "mps" if torch.backends.mps.is_available(
) else "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
labels = ['tức giận', 'ghê tởm', 'so', 'vui vẻ',
          'buồn', 'bất ngờ', 'tự nhiên', 'khinh miệt']


class EmotionRecognition:
    def __init__(self):
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=8, bias=True),
            nn.Softmax(-1)
        )
        self.model.load_state_dict(torch.load(
            'models/best_mobilenetv2.pth', map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, image):
        image = Image.fromarray(image).convert('RGB')
        image_trans = transform(image)
        image_trans = image_trans.unsqueeze(0).to(device)
        predictions = self.model(image_trans)
        predicted_label = torch.argmax(predictions, dim=1).item()

        return unidecode(labels[predicted_label]),predictions[0][predicted_label]
