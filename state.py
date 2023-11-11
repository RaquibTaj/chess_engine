import chess

class State(object):
    def __init__(self):
        self.board = chess.Board()
    
    def serialize(self):
        #257 bits
        pass

    def edges(self):
        return list(self.board.legal_moves)

    def value(self):
        #TODO: add neural net here...
        return 1    #all positions are equal on the board. 
        
if __name__ == "__main__":
    s = State()
    for element in s.edges():
        print(element)
