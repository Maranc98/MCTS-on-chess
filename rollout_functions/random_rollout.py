import random
import numpy as np
from copy import deepcopy

class RandomRollout():
    def __init__(self, max_moves=1000, repeat_n=1):
        self.max_moves = max_moves
        self.repeat_n = repeat_n

    def __call__(self, env):
        rewards = []

        for i in range(self.repeat_n):
            rollout_env = deepcopy(env)
            for j in range(self.max_moves):
                move = random.choice(rollout_env.legal_moves)
                _, reward, done, _ = rollout_env.step(move)
                if done:
                    break
            rewards += [reward]
            rollout_env.close()
            del rollout_env

        return np.mean(rewards)