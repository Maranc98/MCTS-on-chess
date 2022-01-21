import torch

tiles = '.rnbqkpRNBKQP'
tile_to_id = {c:i for i,c in enumerate(tiles)}

class BoardEval():

    def __init__(self, model, ckpt_path):
        self.model = model.load_from_checkpoint(ckpt_path)

    def __call__(self, env):
        board = str(env._board)
        board = [[tile_to_id[c] for c in row if c in tile_to_id] for row in board.split('\n')]
        with torch.no_grad():
            board = torch.tensor([board])
            score = self.model(board)
        
        score = score.detach().numpy()

        return score