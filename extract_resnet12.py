import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import scipy.io as sio
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

# --- ĐỊNH NGHĨA RESNET-12 ---
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.shortcut = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet12(nn.Module):
    def __init__(self, channels):
        super(ResNet12, self).__init__()
        self.inplanes = 3
        self.layer1 = BasicBlock(self.inplanes, channels[0])
        self.layer2 = BasicBlock(channels[0], channels[1])
        self.layer3 = BasicBlock(channels[1], channels[2])
        self.layer4 = BasicBlock(channels[2], channels[3])
        self.keep_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.max_pool2d(self.layer1(x), 2)
        x = F.max_pool2d(self.layer2(x), 2)
        x = F.max_pool2d(self.layer3(x), 2)
        x = F.max_pool2d(self.layer4(x), 2)
        x = self.keep_avg_pool(x)
        return x.view(x.size(0), -1)

# --- CẤU HÌNH ---
IMAGE_DIR = "/content/images" # Bạn hãy đảm bảo đường dẫn ảnh này đúng trên Colab
SPLIT_JSON = "/content/data/full_split.json"
OUTPUT_DIR = "/content/duno"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Khởi tạo model ResNet-12
model = ResNet12([64, 128, 256, 512]).to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(92),
    transforms.CenterCrop(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(SPLIT_JSON, 'r') as f:
        splits = json.load(f)

    all_features, all_labels, train_indices, test_indices = [], [], [], []
    node_id = 0

    for class_name in os.listdir(IMAGE_DIR):
        class_path = os.path.join(IMAGE_DIR, class_name)
        if not os.path.isdir(class_path): continue
        
        print(f"Extracting: {class_name}")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = model(img_t).cpu().numpy().flatten()
                
                all_features.append(feat)
                all_labels.append(class_name)
                
                if class_name in splits['train']:
                    train_indices.append(node_id)
                elif class_name in splits['val']:
                    test_indices.append(node_id)
                node_id += 1
            except: continue

    all_features = np.array(all_features)
    
    # Tạo đồ thị k-NN (k=5)
    nbrs = NearestNeighbors(n_neighbors=6).fit(all_features)
    _, indices = nbrs.kneighbors(all_features)
    with open(os.path.join(OUTPUT_DIR, "tlu_network"), "w") as f:
        for i, neighbors in enumerate(indices):
            for n in neighbors[1:]: f.write(f"{i}\t{n}\n")

    # Lưu file .mat
    def save(idx_list, name):
        sio.savemat(os.path.join(OUTPUT_DIR, name), {
            "Index": np.array(idx_list).reshape(-1, 1),
            "Label": np.array([all_labels[i] for i in idx_list], dtype=object).reshape(-1, 1),
            "Attributes": sparse.csr_matrix(all_features[idx_list])
        })

    save(train_indices, "tlu_train.mat")
    save(test_indices, "tlu_test.mat")
    print("Done!")

if __name__ == "__main__":
    extract()
