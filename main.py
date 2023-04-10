import enum
import threading
import time

import pyglet
import typer
from nes_py.app.play_human import play_human
from nes_py.wrappers import JoypadSpace
from pynput import keyboard
from tqdm import tqdm

from arkanoid import Arkanoid
from heuristic import HeuristicAgent
from terminal import Terminal

ACTIONS = [["NOOP"], ["left"], ["right"], ["A"]]


def key_to_action(keys):
    keymap = {
        pyglet.window.key.S: 0,  # NOOP
        pyglet.window.key.A: 1,  # LEFT
        pyglet.window.key.D: 2,  # RIGHT
        pyglet.window.key.W: 3,  # A
    }
    try:
        return keymap[keys[0]]
    except:
        return 0


def main(render: bool = True, fps: int = 1000, episodes: int = 3, frames: int = 1000):
    terminal = Terminal()
    env = JoypadSpace(Arkanoid(render), ACTIONS)
    agent = HeuristicAgent(env)

    # Set a flag to pause/unpause the game and one to control it
    # This runs in a separate non-blocking thread
    paused = threading.Event()
    human = threading.Event()

    def on_press(key):
        try:
            if key.char == "p":
                paused.clear() if paused.is_set() else paused.set()
            if key.char == "h":
                human.clear() if human.is_set() else human.set()
        except AttributeError:
            pass

    keyboard.Listener(on_press=on_press).start()

    episodes_finished = 0

    try:
        screen, info = env.reset()

        for frame in tqdm(range(frames), position=1, ncols=60):
            while paused.is_set():
                continue

            terminal.startframe()
            if episodes_finished == episodes:
                break

            if human.is_set():
                action = key_to_action(env.env.viewer.pressed_keys)
            else:
                action = agent.get_action(screen, info)

            next_screen, reward, done, next_info = env.step(action)

            if not human.is_set():
                agent.update(screen, info, action, reward, done, next_screen, next_info)

            screen = next_screen
            info = next_info

            if done:
                screen, info = env.reset()
                episodes_finished += 1
                continue

            # Print valuable info
            display = {}
            display["action"] = ACTIONS[action][0]
            display["score"] = info["score"]
            display["level"] = info["level"]
            display["remaining_lives"] = info["remaining_lives"]
            if info["capsule"]["type"] != "None":
                display["capsule"] = info["capsule"]["type"]

            terminal.writedict(display)
            if render:
                env.render()

            if human.is_set():
                time.sleep(0.005)
            else:
                time.sleep(1 / fps)
            terminal.endframe()

    except KeyboardInterrupt:
        env.close()
    finally:
        terminal.close()


if __name__ == "__main__":
    typer.run(main)
