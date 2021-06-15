"""
Credit to: https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""
import torch
import torch.nn as nn

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FashionCNN(pl.LightningModule):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        model = FashionCNN()
        logits = model(x)
        loss = nn.CrossEntropyLoss(logits, y)
        return pl.TrainResult(loss)
    
    def validation_step(self, batch, batch_idx):
        x, y, = batch
        model = FashionCNN()
        logits = model(x)
        val_loss = nn.CrossEntropyLoss(logits, y)
        result = pl.EvalResult()
        result.log('val_loss', val_loss)
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

        
if __name__ == "__main__":
    train_set = FashionMNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=100)
    trainer = pl.Trainer()
    model = FashionCNN()

    trainer.fit(model, train_loader)