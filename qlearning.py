import random
from collections import defaultdict
from agent import ArkAgent
import numpy as np

def state(info):
    state = [
        # info['vaus']['vaus_middle_grid'],
        # info['ball']['ball_high'],
        info['ball']['ball_side'],
        # info['bricks']['bricks_row_bool'],
    ]
    return str(state)

class QLearningAgent(ArkAgent):
    def __init__(
        self,
        env,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate_start: float = 0.95,
        exploration_rate_end: float = 0.05,
        exploration_rate_decay: float = 50000,
    ):
        self.env = env
        self.steps = 0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate_start = exploration_rate_start
        self.exploration_rate_end = exploration_rate_end
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate = exploration_rate_start
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, _screen, info):
        self.steps += 1
        self.exploration_rate = self.exploration_rate_end + (
            self.exploration_rate_start - self.exploration_rate_end
        ) * np.exp(-1.0 * self.steps / self.exploration_rate_decay)

        if np.random.rand() < self.exploration_rate:
            action = self.env.action_space.sample()
        else:
            q_values = self.q_table[state(info)]
            max_q = np.max(q_values)
            idx = np.where(q_values == max_q)[0]
            action = np.random.choice(idx)
        return action

    def update(self, screen, info, action, reward, done, next_screen, next_info):
        q_value = self.q_table[state(info)][action]
        max_q_value = np.max(self.q_table[state(next_info)])

        new_q_value = (1 - self.learning_rate) * q_value + self.learning_rate * (
            reward + self.discount_factor * max_q_value
        )

        self.q_table[state(info)][action] = new_q_value
