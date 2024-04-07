import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
from EffNet.model import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CFG:
    s = 1e-4
    SRtrain=True

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['digit_sum'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(f"./umnist_iccv_1024/train/{file_name}.jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(f"./umnist_iccv_1024/train/{file_name}.jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_optimizer(model):
    optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    return optimizer

def get_scheduler(optimizer):
    scheduler = StepLR(optimizer, step_size=50, gamma=1/3, verbose=True)
    return scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train() # switch to training mode
    running_loss = 0
    count = 0
    for (images, labels) in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        y_preds = model(images)
        
        loss = criterion(y_preds, labels)
        running_loss += loss.item()*labels.shape[0]
        count += 1
        
        loss.backward()
        if CFG.SRtrain:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.grad.data.add_(CFG.s*torch.sign(m.weight.data))  # L1
        optimizer.step()
        optimizer.zero_grad()
        
    return running_loss/count

def valid_fn(valid_loader, model, criterion, device):
    model.eval() # switch to evaluation mode
    preds = []
    running_loss = 0
    count = 0
    
    for (images, labels) in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        running_loss += loss.item()*labels.shape[0]
        count += 1
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
    predictions = np.concatenate(preds)
    
    return (running_loss/count), predictions

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

def train_loop(model_name, model_path, num_epochs):
    train = pd.read_csv('./umnist_iccv_1024/train.csv')
    X = train.drop(['digit_sum'], axis=1)
    y = train['digit_sum']
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
    train_data = pd.merge(x_train, y_train, right_index=True, left_index=True)
    val_data = pd.merge(x_val, y_val, right_index=True, left_index=True)
    
    # create dataset
    train_dataset = TrainDataset(train_data, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(val_data, transform=get_transforms(data='valid'))

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=False, 
                              num_workers=2, pin_memory=True, drop_last=False)

    # create model and transfer to device
    model = EfficientNet.from_pretrained(model_name=model_name, num_classes=28)
    model.to(device)
    
    # select optimizer, scheduler and criterion
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = get_criterion()

    best_score = -1.0
    best_loss = np.inf
    
    # start training
    for epoch in range(num_epochs):
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        # validation
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        valid_labels = val_data['digit_sum']
        
        scheduler.step()

        # scoring
        score = get_score(valid_labels, preds.argmax(1))
        print("score: ", score)

        # code for saving the best model
        if score > best_score:
            print('Score Improved')
            best_score = score
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f}')
            torch.save(model.state_dict(), model_path+f'SR_{model_name}_best_512.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Efficient Net model name")
    parser.add_argument("--epochs", help="Number of epochs")

    args = parser.parse_args()

    # train_loop(model_name='efficientnet-b3', model_path='./umnist_iccv_1024/', num_epochs=100)
    train_loop(model_name=args.model_name, model_path='./umnist_iccv_1024/', num_epochs=args.epochs)
    torch.cuda.empty_cache()