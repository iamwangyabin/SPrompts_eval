import os
import argparse
from PIL import Image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from utils.core50_data_loader import CORE50
from utils.toolkit import accuracy_binary, accuracy_domain, accuracy_core50

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--resume', type=str, default='', help='resume model')
    parser.add_argument('--dataroot', type=str, default='/home/wangyabin/workspace/DeepFake_Data/CL_data/', help='data path')
    parser.add_argument('--datatype', type=str, default='core50', help='data type')
    return parser

class DummyDataset(Dataset):
    def __init__(self, data_path, data_type):

        self.trsf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        images = []
        labels = []
        if data_type == "deepfake":
            subsets = ["gaugan", "biggan", "wild", "whichfaceisreal", "san"]
            multiclass = [0,0,0,0,0]
            for id, name in enumerate(subsets):
                root_ = os.path.join(data_path, name, 'val')
                # sub_classes = ['']
                sub_classes = os.listdir(root_) if multiclass[id] else ['']
                for cls in sub_classes:
                    for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                        images.append(os.path.join(root_, cls, '0_real', imgname))
                        labels.append(0 + 2 * id)

                    for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                        images.append(os.path.join(root_, cls, '1_fake', imgname))
                        labels.append(1 + 2 * id)
        elif data_type == "domainnet":
            self.data_root = data_path
            self.image_list_root = self.data_root
            self.domain_names = ["clipart","infograph","painting","quickdraw", "real","sketch",]
            image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
            imgs = []
            for image_list_path in image_list_paths:
                image_list = open(image_list_path).readlines()
                imgs += [(val.split()[0], int(val.split()[1])) for val in image_list]

            for item in imgs:
                images.append(os.path.join(self.data_root, item[0]))
                labels.append(item[1])
        elif data_type == "core50":
            self.dataset_generator = CORE50(root=data_path, scenario="ni")
            images, labels = self.dataset_generator.get_test_set()
            labels = labels.tolist()
        else:
            pass

        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.trsf(self.pil_loader(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

args = setup_parser().parse_args()
model = torch.load(args.resume)
device = "cuda:0"
model = model.to(device)
test_dataset = DummyDataset(args.dataroot, args.datatype)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

X,Y = [], []

for id, task_centers in enumerate(model.all_keys):
    X.append(task_centers.detach().cpu().numpy())
    Y.append(np.array([id]*len(task_centers)))

X = np.concatenate(X,0)
Y = np.concatenate(Y,0)
neigh = KNeighborsClassifier(n_neighbors=1, metric='l1')
neigh.fit(X, Y)

selectionsss = []
from collections import Counter

y_pred, y_true = [], []
for _, (path, inputs, targets) in enumerate(test_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        feature = model.extract_vector(inputs)
        selection = neigh.predict(feature.detach().cpu().numpy())
        # selectionsss.extend(selection)
        selection = torch.tensor(selection).to(device)
        outputs = model.interface(inputs, selection)
    predicts = torch.topk(outputs, k=2, dim=1, largest=True, sorted=True)[1]
    y_pred.append(predicts.cpu().numpy())
    y_true.append(targets.cpu().numpy())

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

result = Counter(selectionsss)
print(result)
if args.datatype == 'deepfake':
    print(accuracy_binary(y_pred.T[0], y_true))
elif args.datatype == 'domainnet':
    print(accuracy_domain(y_pred.T[0], y_true))
elif args.datatype == 'core50':
    print(accuracy_core50(y_pred.T[0], y_true))


