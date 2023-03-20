from arkanoid import Arkanoid
import time
from nes_py.wrappers import JoypadSpace


actions = [["NOOP"], ["left"], ["right"], ["A"]]

ark = Arkanoid()
ark = JoypadSpace(ark, actions)

import numpy as np

done = True
for i in range(1000):
    if done:
        _ = ark.reset()
        done = False

    action = ark.action_space.sample()

    state, reward, done, info = ark.step(action)
    print(f"{info}")

    ark.render()
    time.sleep(0.01)
ark.close()
