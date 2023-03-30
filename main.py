import time
import numpy as np
import typer
from nes_py.wrappers import JoypadSpace
from nes_py.app.play_human import play_human
from tqdm import tqdm
from arkanoid import Arkanoid
from terminal import Terminal
from pynput import keyboard
import enum
import threading

class Mode(str, enum.Enum):
    ia = "ia"
    human = "human"

def main(mode: Mode = Mode.ia,
         render: bool = True,
         fps: int = 1000,
         episodes: int = 3,
         frames: int = 1000):
    actions = [["NOOP"], ["left"], ["right"], ["A"]]

    terminal = Terminal()
    ark = JoypadSpace(Arkanoid(), actions)

    # Set a flag to pause/unpause the game
    paused = threading.Event()
    def on_press(key):
        try:
            if (key.char == "p"):
                paused.clear() if paused.is_set() else paused.set()
        except AttributeError:
            pass

    keyboard.Listener(on_press=on_press).start()

    shot_laser = 1
    episodes_finished = 0
    action = 0

    if mode == "human":
        play_human(ark)
    else:
        try:
            for i in tqdm(range(frames), position=1, ncols=60):
                while paused.is_set():
                    continue

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
