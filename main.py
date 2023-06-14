import enum
import threading
import time
from typing import Literal
from itertools import count
import pyglet
import typer
from nes_py.app.play_human import play_human
from nes_py.wrappers import JoypadSpace
from pynput import keyboard
from tqdm import tqdm

from arkanoid import Arkanoid
from dqn import DQNAgent
from heuristic import HeuristicAgent
from terminal import Terminal

from typing import Optional
import pathlib

ACTIONS = [["NOOP"], ["left"], ["right"]]

import enum

app = typer.Typer()


class Agent(enum.Enum):
    heuristic = "heuristic"
    dqn = "dqn"
    human = "human"


def key_to_action(keys):
    keymap = {
        # pyglet.window.key.S: 0,  # NOOP
        pyglet.window.key.A: 1,  # LEFT
        pyglet.window.key.D: 2,  # RIGHT
        # pyglet.window.key.W: 3,  # A
    }
    try:
        return keymap[keys[0]]
    except:
        return 0


@app.command()
def play(
    render: bool = True,
    fps: int = 50, # Set to 0 to go as quick as possible
    episodes: int = 3,
    frames: Optional[int] = None,
    agent: Agent = Agent.heuristic.value,
    checkpoint_dir: Optional[pathlib.Path] = None,
):
    terminal = Terminal()
    env = JoypadSpace(Arkanoid(render), ACTIONS)

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

    if agent == Agent.heuristic:
        agent = HeuristicAgent(env)
    elif agent == Agent.dqn:
        if checkpoint_dir is None:
            raise ValueError(
                "When using DQN Agent, you need to specify where to load the checkpoint from"
            )
        agent = DQNAgent.load(env, checkpoint_dir)
        print("Loaded agent from checkpoint")
    elif agent == Agent.human:
        agent = HeuristicAgent(env)
        human.set()

    episodes_finished = 0

    scores = []
    try:
        screen, info = env.reset()

        for frame in tqdm(range(frames) if frames else count(), position=1, ncols=60):
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
                scores.append(info['score'])
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
            display["hit_counter"] = info["hit_counter"]

            terminal.write(f"Episode Scores: {scores}\n")
            terminal.writedict(display)
            if render:
                env.render()
            if fps:
                time.sleep(1 / fps)

            terminal.endframe()

    except KeyboardInterrupt:
        env.close()
    finally:
        terminal.close()


import pathlib


@app.command()
def train(
    episodes: int = 1000,
    batch_size: int = 128,
    checkpoint_dir: pathlib.Path = "checkpoints",
    save_every: Optional[int] = None,
    resume: bool = False,
):
    env = JoypadSpace(Arkanoid(), ACTIONS)
    DQNAgent.train(
        env,
        checkpoint_dir,
        batch_size=batch_size,
        episodes=episodes,
        save_every=save_every,
        resume=resume,
    )


if __name__ == "__main__":
    app()
