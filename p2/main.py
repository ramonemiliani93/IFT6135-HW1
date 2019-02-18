import os
import os.path as osp
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

if not osp.isdir('data'):
    os.mkdir('data')


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.classify = nn.Sequential(
            nn.Linear(3200, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.classify(out)
        return out


train_dataset = datasets.MNIST('data', download=True, transform=transforms.ToTensor())

val_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

model = CNN()
optim = torch.optim.SGD(model.parameters(), lr=1e-2)
pytorch_total_params = sum(p.numel() for p in model.parameters())
criterion = nn.CrossEntropyLoss()


epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
epoch_val_acc = []
for epoch in range(10):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    model.train()
    for batch_idx, (batch_ims, batch_ys) in enumerate(train_dataloader):
        optim.zero_grad()
        out = model(batch_ims)
        preds = torch.argmax(out, dim=-1)

        train_acc.append((preds == batch_ys).sum().item())
        loss = criterion(out, batch_ys)
        train_loss.append(loss.item()*64)
        loss.backward()
        optim.step()

    model.eval()
    for batch_idx, (batch_ims, batch_ys) in enumerate(val_dataloader):
        out = model(batch_ims)
        loss = criterion(out, batch_ys)
        val_loss.append(loss.item() * 64)
        preds = torch.argmax(out, dim=-1)
        val_acc.append((preds == batch_ys).sum().item())

    epoch_train_loss.append(sum(train_loss))
    epoch_train_acc.append(sum(train_acc)/60000)
    epoch_val_loss.append(sum(val_loss))
    epoch_val_acc.append(sum(val_acc)/10000)

epoch_train_loss = np.array(epoch_train_loss)
epoch_val_loss = np.array(epoch_val_loss)

epoch_train_loss /= 60000
epoch_val_loss /= 10000

plt.plot(np.arange(0, 10), epoch_train_loss, label='train')
plt.plot(np.arange(0, 10), epoch_val_loss, label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('cnn_loss.png')
plt.show()


