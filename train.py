from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
from torch import optim
import torch
import torch.nn.functional as F

class chess_value_dataset(Dataset):
    def __init__(self):
        dat = np.load("processed/dataset_10M.npz")
        self.X = dat['x']
        self.Y = dat['y']
        print("loaded", self.X.shape, self.Y.shape)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)

        self.last = nn.Linear(128, 1)

    def forward(self, x):
        #8x8
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        #4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        #2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        #1x128
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)
        x = self.last(x)

        #value o/p
        return F.tanh(x) # tanh is ideal activation funtion cause [-1,+1] 
    
if __name__ == "__main__":    
    chess_dataset = chess_value_dataset()
    model = Net()
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=256, shuffle = True)
    optimizer = optim.Adam(model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Setting device based on availability 
    model.cuda() #TODO: make a function to dynamically assign device based on availabilty 
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(100):
        total_loss = 0
        div_factor = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            #data = data.unsqueeze_(-1) -> previusly used to manually coorect the PGN files while training (Bad Idea)
            target = target.unsqueeze(-1)
            data = data.float()
            target = target.float()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            div_factor += 1

        print(f"Epoch: {epoch} Batch: {batch_idx} Loss: {total_loss/div_factor}") #printing the loss of the network at every forward and backward pass


