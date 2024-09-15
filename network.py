import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, numClasses, hash_code):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        #self.bn1 = nn.BatchNorm2d(4096)
        self.fc4 = nn.Linear(1024, numClasses)
        self.fc5 = nn.Linear(1024, hash_code)
        self.tanh = nn.Tanh()
        #self.BN1 = nn.BatchNorm1d(hash_code, momentum=0.1)
        

    def forward(self, x):
        #print(x)
        x = F.relu(self.conv1(x))
        x, indices1 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = F.relu(self.conv2(x))
        x, indices2 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x, indices3 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)
        x = x.view(x.size(0), 256 * 6 * 6)
        #x = F.dropout(x)
        #x = self.bn1(x)
        #print(x.shape)
        x = self.fc1(x)
        #x_fc6 = x
        #x = F.dropout(x)
        x = self.fc2(x)
        x = F.relu(self.fc3(x))#x
        #x = F.relu(self.fc3(x))
        #x = F.dropout(x)
        x1 = self.fc4(x)
        h = self.fc5(x)
        h = self.tanh(h) #apply tanh(beta*x)
        #h = self.BN1(h)
        return x1, h  

  

