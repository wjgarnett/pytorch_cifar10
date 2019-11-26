#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net1st(nn.Module):

    def __init__(self):
        super(Net1st, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 二维卷积，in_channels:3, out_channels:6, kernerl_size:5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)  #不同batch_size都能适应
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x