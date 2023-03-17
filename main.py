from arkanoid import Arkanoid

ark = Arkanoid()
done = True
for i in range(1000):
    if done:
        _ = ark.reset()
    action = ark.action_space.sample()
    _, reward, done, info = ark.step(action)
    ark.render()
ark.close()
