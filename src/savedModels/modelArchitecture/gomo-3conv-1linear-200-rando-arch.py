import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from IPython.display import display
import pandas as pd
import time 

class DQN(nn.Module):
    
    def __init__(self, outputs):
        super(DQN, self).__init__()
        # 6 by 7, 10 by 11 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7,7), padding=0, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=0, stride=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(3,3), padding=0, stride=1)
        self.pool = nn.MaxPool2d((5,5))
        self.MLP1 = nn.Linear(256, 256)
        self.MLP4 = nn.Linear(256, outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.pool(x)
        # flatten the feature vector except batch dimension
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.MLP1(x))
        return self.MLP4(x)


def get_Model():
    PATH = "src/savedModels/gomo-3conv-1linear_200-rando.pth"
    model = DQN(15 **2)
    model.load_state_dict(torch.load(PATH))
    return model



if __name__ == "__main__": 
    players = ['p1','p2']
    model = get_Model()
