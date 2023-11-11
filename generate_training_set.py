import chess.pgn
import os
from state import State

#print(os.listdir("Dataset/"))

Path = "Dataset/" #Path to the pgn files.

for file in os.listdir("Dataset"):
    pgn = open(os.path.join("Dataset/", file))
    while 1:
        try:
            game = chess.pgn.read_game(pgn)
        except Exception:
            break;
        board = game.board()
        value = {'1/2-1/2':0, '0-1':-1,'1-0':1}[game.headers['Result']]
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            # TODO: extract the boards
            print(value)
            print(State(board).serialize()[:,:, 0]) #Need to convert FEN to Vector representation
    break