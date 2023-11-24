import chess.pgn
import os
from state import State
import numpy as np


def get_dataset(num_samples=None):
    X,Y = [], []
    gn = 0
    for file in os.listdir("Dataset"):
        pgn = open(os.path.join("Dataset/", file))
        while 1:
            try:
                game = chess.pgn.read_game(pgn) #pgn stands for Portable Game Notation
            except Exception:
                break;
            
            board = game.board()
            value = {'1/2-1/2':0, '0-1':-1,'1-0':1}[game.headers['Result']]
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                ser = State(board).serialize()[:,:, 0]
                X.append(ser)
                Y.append(value)
            print("Parsing game %d, got %d examples" % (gn, len(X)))
            if num_samples is not None and len(X) > num_samples:
                return X, Y  #returning X and Y if we have reached the number of samples required by the user
            gn += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == "__main__":
    X,Y = get_dataset(9.6*1e6)
    np.savez("processed/dataset_all.npz", x=X, y=Y)
