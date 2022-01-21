import gym
import gym_chess
import anytree as at
from anytree.exporter import DotExporter
import numpy as np
import random
import time
from copy import deepcopy
from players.stockfish_player import StockfishPlayer

from rollout_functions import RandomRollout, BoardEval
from players import RandomPlayer, StockfishPlayer
from models import BoardScoreGenerator

class MCTS():
    def __init__(
        self,
        rollout_fn=None,
        show_text=False,
        show_tree=False,
        save_tree=False,
        exploration_c=2,
        max_steps=1000,
        max_time=10,
        ):

        self.s0 = None
        self.env = None
        self.rollout_fn = rollout_fn if rollout_fn is not None else RandomRollout()

        self.show_text = show_text
        self.show_tree = show_tree
        self.save_tree = save_tree
        self.exploration_c = exploration_c
        self.max_steps = max_steps
        self.max_time = max_time

        self.tree = at.Node("S0", t=0, n=0)
        self.curr_node = self.tree
        self.moves_log = []
        self.step_counter = 0
        self.node_counter = 0

    def step(self):
        self.curr_node = self.tree
        self.reset_env()

        # Get UCB1 scores for children nodes and go down the best UCB1 until leaf
        while self.curr_node.children:
            scores = [self.ucb1(child) for child in self.curr_node.children]
            self.set_curr_node(np.argmax(scores))

        if self.curr_node.n == 0:
            v = self.rollout()
        else:
            self.curr_node.children = [at.Node("", t=0, n=0, action=a, opponent_action=None) for a in self.env.legal_moves]
            for child in self.curr_node.children:
                self.node_counter += 1
                child.name = f"S{self.node_counter}"
            self.set_curr_node(random.randrange(len(self.curr_node.children)))
            v = self.rollout()

        self.backpropagate(v)

        if self.show_tree:
            print(at.RenderTree(self.tree))
        if self.save_tree:
            DotExporter(
                self.tree,
                nodeattrfunc=lambda n: f"label=\"{n.name}\nt: {n.t}\nn: {n.n}\"",
                edgeattrfunc=lambda n, c: f"label=\"{c.action}, {c.opponent_action}\"",
            ).to_dotfile(f"trees/{len(self.moves_log)}_{self.step_counter:05d}.dot")
            self.step_counter += 1

    def set_curr_node(self, i):
        self.curr_node = self.curr_node.children[i]
        self.env.step(self.curr_node.action)
        if self.curr_node.opponent_action is None:
            self.curr_node.opponent_action = random.choice(self.env.legal_moves)
        self.env.step(self.curr_node.opponent_action)

    def rollout(self):
        reward = self.rollout_fn(self.env)

        if self.show_text:
            print(f"Reward {reward}")
        return reward

    def backpropagate(self, v):
        self.curr_node.t += v
        self.curr_node.n += 1
        while self.curr_node.parent:
            self.curr_node = self.curr_node.parent
            self.curr_node.t += v
            self.curr_node.n += 1

    def ucb1(self, node):
        if node.n == 0:
            return np.Inf

        reward_avg = node.t / node.n
        exploration_score = np.sqrt(np.log(self.tree.n) / node.n)
        return reward_avg + self.exploration_c * exploration_score

    def set_state(self, state):
        del self.s0
        self.tree = at.Node("S0", t=0, n=0)
        self.s0 = deepcopy(state)

    def reset_env(self):
        del self.env
        self.env = deepcopy(self.s0)

    def get_best_move(self):
        t0 = time.time()
        for i in range(self.max_steps):
            self.step()
            if self.max_time is not None and (time.time() - t0) > self.max_time:
                break

        # Choose best action as the action with the most simulations
        best_child = np.argmax([child.n for child in self.tree.children])
        best_move = self.tree.children[best_child].action

        if self.show_text:
            print(f"Found move {best_move}\nTime elapsed: {time.time() - t0}\nSteps completed: {i+1}")
        return best_move

    def add_move(self, move):
        self.moves_log += [move]

        del self.tree
        self.tree = at.Node("", t=0, n=0)
        self.step_counter = 0
        self.node_counter = 0

if __name__ == "__main__":
    player1 = MCTS(
        #rollout_fn=RandomRollout(repeat_n=2),
        rollout_fn=BoardEval(BoardScoreGenerator, 'ckpts/eval_model.ckpt'),
        max_steps=1000,
        max_time=5,
        show_tree=False,
        show_text=False,
        save_tree=False,
    )

    #player2 = RandomPlayer()
    player2 = StockfishPlayer()
    players = [player1, player2]

    env = gym.make('Chess-v0')
    env.reset()

    for i in range(10):
        for player in players:
            player.set_state(env)
            move = player.get_best_move()
            print(move)
            env.step(move)
            print(env.render(), "\n")
