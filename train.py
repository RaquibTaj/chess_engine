import chess.pgn
import os

print(os.listdir("Dataset/"))

Path = "Dataset/"

for file in os.listdir("Dataset"):
    pgn = open(os.path.join("Dataset/", file))
    while 1:
        try:
            game = chess.pgn.read_game(pgn)
        except Exception:
            break;
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            print(board)
        exit (0)
    break