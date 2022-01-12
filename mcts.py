import gym
import gym_chess
import anytree as at
from anytree.exporter import DotExporter
import numpy as np
import random
import time

class MCTS():
    def __init__(
        self,
        env,
        show_text=False,
        show_tree=False,
        save_tree=False,
        exploration_c=2,
        ):

        self.env = env
        self.show_text = show_text
        self.show_tree = show_tree
        self.save_tree = save_tree
        self.exploration_c = exploration_c

        self.tree = at.Node("S0", t=0, n=0)
        self.curr_node = self.tree
        self.moves_log = []
        self.step_counter = 0
        self.node_counter = 0

    def step(self):
        self.curr_node = self.tree
        self.reset_env()

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
        done = False
        for i in range(1000):
            self.env.render()
            move = random.choice(self.env.legal_moves)
            observation, reward, done, info = self.env.step(move)
            if done:
                break
        self.env.close()

        if self.show_text:
            print(f"MAX STEPS: {i} got {reward}")
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

    def reset_env(self):
        self.env.reset()
        for move in self.moves_log:
            self.env.step(move)

    def get_best_move(self, steps=None, max_time=None):
        steps = steps if steps is not None else 1000
        t0 = time.time()
        for i in range(steps):
            self.step()
            if max_time is not None and (time.time() - t0) > max_time:
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
    solver = MCTS(gym.make('Chess-v0'), show_tree=False, show_text=False, save_tree=False)

    outside_env = gym.make('Chess-v0')
    outside_env.reset()

    for i in range(10):
        move = solver.get_best_move(steps=1000, max_time=10)
        outside_env.step(move)
        solver.add_move(move)
        print(outside_env.render(), "\n")

        opponent_move = random.choice(outside_env.legal_moves)
        outside_env.step(opponent_move)
        solver.add_move(opponent_move)
        print(outside_env.render(), "\n\n")
