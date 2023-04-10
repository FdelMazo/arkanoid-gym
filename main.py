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
import pyglet

ACTIONS = [["NOOP"], ["left"], ["right"], ["A"]]

def key_to_action(keys):
    keymap = {
        pyglet.window.key.S: 0, # NOOP
        pyglet.window.key.A: 1, # LEFT
        pyglet.window.key.D: 2, # RIGHT
        pyglet.window.key.W: 3  # A
    }
    try:
        return keymap[keys[0]]
    except:
        return 0

def main(render: bool = True,
         fps: int = 1000,
         episodes: int = 3,
         frames: int = 1000):

    terminal = Terminal()
    ark = JoypadSpace(Arkanoid(render), ACTIONS)

    # Set a flag to pause/unpause the game and one to control it
    # This runs in a separate non-blocking thread
    paused = threading.Event()
    human = threading.Event()
    def on_press(key):
        try:
            if (key.char == "p"):
                paused.clear() if paused.is_set() else paused.set()
            if (key.char == "h"):
                human.clear() if human.is_set() else human.set()
        except AttributeError:
            pass

    keyboard.Listener(on_press=on_press).start()

    shot_laser = 1
    episodes_finished = 0
    action = 0

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

            if human.is_set():
                action = key_to_action(ark.env.viewer.pressed_keys)
            else:
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
            display["action"] = ACTIONS[action][0]
            display["score"] = info["score"]
            display["level"] = info["level"]
            display["remaining_lives"] = info["remaining_lives"]
            if (info["capsule"]["type"] != "None"):
                display["capsule"] = info["capsule"]["type"]

            terminal.writedict(display)
            if render:
                ark.render()

            if human.is_set():
                time.sleep(0.005)
            else:
                time.sleep(1/fps)
            terminal.endframe()

    except KeyboardInterrupt:
        ark.close()
    finally:
        terminal.close()

if __name__ == "__main__":
    typer.run(main)
