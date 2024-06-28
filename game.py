import torch
from state import State 
from train import Net

if __name__ == "__main__":
    vals = torch.load("saved_model/weights.pth", map_location=lambda storage, loc: storage)
    model = Net()
    model.load_state_dict(vals)
    s = State()
    board = s.serialize()[None]
    output =model(torch.tensor(board).float())
    print(output)