import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        
        self.flat_size = 128 * 16 * 16 

        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x