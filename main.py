from arkanoid import Arkanoid
import time

ark = Arkanoid()
done = True
for i in range(1000):
    if done:
        _ = ark.reset()
    action = ark.action_space.sample()
    state, reward, done, info = ark.step(action)
    ark.render()
ark.close()
