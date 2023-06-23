"""Heuristic agent.

Tries to stay below the ball at all times."""

import numpy as np

from agent import ArkAgent


class HeuristicAgent(ArkAgent):
    def __init__(self, env):
        self.env = env
        self.shot_laser = 1

    def get_action(self, _screen, info) -> int:
        if info["vaus"]["vaus_status_string"] == "laser":
            self.shot_laser += 1
        if self.shot_laser % 2 == 0 and info["vaus"]["vaus_status_string"] == "laser":
            action = 3
        if info["ball"]["ball_x"] < info["vaus"]["vaus_left_x"]:
            action = 1
        elif info["ball"]["ball_x"] > info["vaus"]["vaus_right_x"]:
            action = 2
        else:
            action = self.env.action_space.sample(
                mask=np.array([1, 0, 0, 1], dtype=np.int8)
            )
        return action

    def update(self, _screen, _info, _action, _reward, _done, _next_screen, _next_info):
        pass
