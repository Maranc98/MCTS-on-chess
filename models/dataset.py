import torch
import chess
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

tiles = '.rnbqkpRNBKQP'
tile_to_id = {c:i for i,c in enumerate(tiles)}

def eval_to_float(x):
    x = x.replace(u'\ufeff','')
    if '#' in x:
        sign = 1 if '+' in x else -1
        x = x.replace('#','').replace('+','').replace('-','')
        return sign*(500 - int(x))
    else:
        try:
            return float(x)
        except:
            print(x)

class BoardEvaluationsDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.data['Evaluation'] = self.data['Evaluation'].apply(eval_to_float)
        self.data['Evaluation'].clip(upper=500, lower=-500, inplace=True)
        self.data['Evaluation'] = self.data['Evaluation'] / 500
        print(np.histogram(self.data['Evaluation']))
        self.data['Evaluation'] = self.data['Evaluation'] / self.data['Evaluation'].max()

        print(self.data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.data['FEN'][idx]
        board_txt = str(chess.Board(fen))
        board = [[tile_to_id[c] for c in row if c in tile_to_id]for row in board_txt.split('\n')]

        score = self.data['Evaluation'][idx]

        return torch.tensor(board, dtype=torch.long).to('cuda'), torch.tensor(score, dtype=torch.float).to('cuda')
