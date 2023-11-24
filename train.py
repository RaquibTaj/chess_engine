from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn

class chess_value_dataset(Dataset):
    def __init__(self):
        dat = np.load("processed/dataset_all.npz")
        self.X = dat['x']
        self.Y = dat['y']
        print("loaded", self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shapr[0]
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'Y': self.Y[idx]}
    
chess_dataset = chess_value_dataset()

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
    
    def forward(self, x):
        #8x8
        x = F.relu(nn.Conv2d(5, 10, kernel_size=3))
        x = F.relu(nn.Conv2d(10, 10, kernel_size=3))
        x = F.relu(nn.Conv2d(10, 10, kernel_size=3))
        x = F.max_pool2d(x)

        #4x4
        x = F.relu(nn.Conv2d(20, 20, kernel_size=3))
        x = F.relu(nn.Conv2d(20, 20, kernel_size=3))
        x = F.relu(nn.Conv2d(20, 20, kernel_size=3))
        x = F.max_pool2d(x)

        #2x2
        x = F.relu(nn.Conv2d(40, 40, kernel_size=3))
        x = F.relu(nn.Conv2d(40, 40, kernel_size=3))
        x = F.relu(nn.Conv2d(40, 40, kernel_size=3))
        x = F.max_pool2d(x)

        #1x1
        x = x.view(-1, 1)

        return F.log_softmax(x, dim=1)