import chess.pgn
import os
from state import State
import numpy as np


def get_dataset(num_samples=None):
    X,Y = [], []
    gn = 0
    values = {'1/2-1/2':0, '0-1':-1,'1-0':1}
    for file in os.listdir("Dataset"):
        pgn = open(os.path.join("Dataset/", file))
        while 1:
            try:
                game = chess.pgn.read_game(pgn) #pgn stands for Portable Game Notation
            except Exception:
                print(Exception)
                break
            
            res = game.headers['Result']
            if res not in values: #if tge PGN files have corrupt data, we ignore the particular example and move on..
                continue

            value = values[res]
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                ser = State(board).serialize()
                X.append(ser)
                Y.append(value)
            print(f"Parsing game {gn}, got {len(X)} examples")
            if num_samples is not None and len(X) > num_samples:
                return X, Y  #returning X and Y if we have reached the number of samples required by the user
            gn += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == "__main__":
    X,Y = get_dataset(1e7)
    np.savez("processed/dataset_10M.npz", x=X, y=Y)
