import time
import pprint

import numpy as np
import typer
import os
import curses
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
from arkanoid import Arkanoid


def main(render: bool = True, fps: int = 1000, episodes: int = 3, frames: int = 10_000):
    # Initialize terminal screen
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    stdscr.keypad(True)

    # Read all the ascii art logos instead of opening the file on each frame
    asciiarts = {}
    for art in os.listdir("./asciiart"):
        with open(f"./asciiart/{art}", 'r') as f:
            width = art.split(".")[0].split("-")[1]
            asciiarts[width] = f.readlines()

    actions = [["NOOP"], ["left"], ["right"], ["A"]]

    ark = Arkanoid()
    ark = JoypadSpace(ark, actions)
    _ = ark.reset()

    shot_laser = 1
    episodes_finished = 0
    action = 0

    for i in tqdm(range(frames), position=1, ncols=68):
        # Clear the terminal screen
        stdscr.erase()

        # Hard limit on the terminal size
        if (stdscr.getmaxyx()[0] < 16 or stdscr.getmaxyx()[1] < 70):
            stdscr.addstr("Terminal too small!!\n")
            cursesenabled = False
        else:
            cursesenabled = True

        # Draw the logo
        asciiart = None
        for width in sorted(asciiarts.keys(), key=int):
            if stdscr.getmaxyx()[1] >= int(width):
                asciiart = asciiarts[width]

        if (cursesenabled and asciiart):
            for i, line in enumerate(asciiart):
                stdscr.addstr(i, 0, line)
            stdscr.addstr("\n\n")

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

        if (cursesenabled):
            stdscr.addstr(f"{pprint.pformat(display, sort_dicts=False, width=70)}\n")

        if render:
            ark.render()

        time.sleep(1 / fps)
        stdscr.refresh()

    ark.close()
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


if __name__ == "__main__":
    typer.run(main)
