import time
import numpy as np
import typer
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
from arkanoid import Arkanoid
from terminal import Terminal

def main(render: bool = True, fps: int = 1000, episodes: int = 3, frames: int = 1000):
    actions = [["NOOP"], ["left"], ["right"], ["A"]]

    terminal = Terminal()
    ark = Arkanoid()
    ark = JoypadSpace(ark, actions)

    shot_laser = 1
    episodes_finished = 0
    action = 0

    try:
        for i in tqdm(range(frames), position=1, ncols=60):
            terminal.startframe()
            if episodes_finished == episodes:
                break

            state, reward, done, info = ark.step(action)

            if done:
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

            # Print valuable info
            display = {}
            display["action"] = actions[action][0]
            display["score"] = info["score"]
            display["level"] = info["level"]
            display["remaining_lives"] = info["remaining_lives"]
            if (info["capsule"]["type"] != "None"):
                display["capsule"] = info["capsule"]["type"]

            terminal.writedict(display)
            if render:
                ark.render()

            time.sleep(1 / fps)
            terminal.endframe()

    except KeyboardInterrupt:
        ark.close()
    finally:
        terminal.close()

if __name__ == "__main__":
    typer.run(main)
