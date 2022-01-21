import chess
from stockfish import Stockfish

class StockfishPlayer():
    def __init__(self, path="./stockfish/stockfish.exe"):
        self.stockfish = Stockfish(path=path)
        self.fen = None

    def set_state(self, env):
        self.fen = env._board.fen()
        self.stockfish.set_fen_position(self.fen)

    def get_best_move(self):
        move = self.stockfish.get_best_move()
        move = chess.Move.from_uci(move)
        return move
