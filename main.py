import time
from pprint import pprint

import numpy as np
import typer
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm

from arkanoid import Arkanoid


def main(render: bool = True, fps: int = 1000, episodes: int = 3, frames: int = 10_000):
    actions = [["NOOP"], ["left"], ["right"], ["A"]]

    ark = Arkanoid()
    ark = JoypadSpace(ark, actions)
    _ = ark.reset()

    shot_laser = 1
    episodes_finished = 0
    action = 0

    for i in tqdm(range(frames)):
        if episodes_finished == episodes:
            break

        state, reward, done, info = ark.step(action)

        if done:
            print("Episode over!")
            _ = ark.reset()
            done = False
            episodes_finished += 1
            continue

        if info["vaus_status"] == "laser":
            shot_laser += 1
        if shot_laser % 2 == 0 and info["vaus_status"] == "laser":
            action = 3
        elif info["ball_x"] < info["vaus_pos"]["vaus_left_x"]:
            action = 1
        elif info["ball_x"] > info["vaus_pos"]["vaus_right_x"]:
            action = 2
        else:
            action = ark.action_space.sample(mask=np.array([1, 0, 0, 1], dtype=np.int8))

        # pprint(info)
        if render:
            ark.render()

        time.sleep(1 / fps)

    ark.close()


if __name__ == "__main__":
    typer.run(main)
