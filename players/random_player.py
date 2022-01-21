import random

class RandomPlayer():

    def __init__(self):
        self.env = None

    def set_state(self, state):
        del self.env
        self.env = state

    def get_best_move(self):
        return random.choice(self.env.legal_moves)