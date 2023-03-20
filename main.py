import time

from nes_py.wrappers import JoypadSpace

from arkanoid import Arkanoid

actions = [["NOOP"], ["left"], ["right"], ["A"]]

ark = Arkanoid()
ark = JoypadSpace(ark, actions)
from pprint import pprint

import numpy as np

done = True
action = 0
shot_laser = 1
for i in range(50_000):
    if done:
        _ = ark.reset()
        done = False

    state, reward, done, info = ark.step(action)
    if info["vaus_status"] == "laser":
        shot_laser += 1

    if shot_laser % 2 == 0 and info["vaus_status"] == "laser":
        action = 3
    elif info["ball_grid_y"] * (168 / 11) + 16 < info["vaus_left_x"]:
        action = 1
    elif (
        info["ball_grid_y"] * (168 / 11)
        + 16
        + info["vaus_very_right_x"]
        - info["vaus_very_left_x"]
        > info["vaus_right_x"]
    ):
        action = 2
    else:
        action = ark.action_space.sample(mask=np.array([1, 0, 0, 1], dtype=np.int8))

    # pprint(info)
    ark.render()
    time.sleep(0.001)
ark.close()
