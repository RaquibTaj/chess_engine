import torch
from state import State 
from train import Net

class Evaluator(object):
    
    def __init__(self):
        vals = torch.load("saved_model/weights.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, s):
        board = s.serialize()[None]
        output = self.model(torch.tensor(board).float())
        return float(output.data[0][0]) 

if __name__ == "__main__":
    e = Evaluator()
    s = State()
    print(e(s))