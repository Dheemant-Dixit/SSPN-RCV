import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from EffNet.model import EfficientNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class TestDataset(Dataset):
    def __init__(self, fnames, transform=None, data_dir=None):
        self.file_names = fnames
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(f"{self.data_dir}/{file_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def test_fn(test_loader, model, device):
    model.eval() # switch to evaluation mode
    preds = []
    for images in tqdm(test_loader):
        images = images.to(device)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
    predictions = np.concatenate(preds)
    
    return predictions.argmax(1)

def pred(data_dir):
    fnames = os.listdir(data_dir)
    test_dataset = TestDataset(fnames, transform=get_transforms(data='valid'), data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                              num_workers=2, pin_memory=True, drop_last=False)
    model = EfficientNet.from_name_pruned('efficientnet-b1', "SR_best_512_3.pth", 
                                          num_classes=28, device=device)
    model.to(device)
    
    predictions = test_fn(test_loader, model, device)
    fnames1 = fnames
    for i in range(len(fnames1)):
        fnames1[i] = fnames1[i][:-5]
    df = pd.DataFrame(list(zip(fnames1, predictions)),columns =['image_id', 'digit_sum'])
    df.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="Data directory")

    args = parser.parse_args()
    pred(args.data_dir)
