import numpy as np


class HeuristicAgent:
    def __init__(self, env):
        self.env = env
        self.shot_laser = 1

    def get_action(self, _screen, info):
        if info["vaus_status"] == "laser":
            self.shot_laser += 1
        if self.shot_laser % 2 == 0 and info["vaus_status"] == "laser":
            action = 3
        elif info["ball_x"] < info["vaus_pos"]["vaus_left_x"]:
            action = 1
        elif info["ball_x"] > info["vaus_pos"]["vaus_right_x"]:
            action = 2
        else:
            action = self.env.action_space.sample(
                mask=np.array([1, 0, 0, 1], dtype=np.int8)
            )
        return action

    def update(self, _screen, _info, _action, _reward, _done, _next_screen, _next_info):
        pass
