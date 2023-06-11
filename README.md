# Arkanoid Gym

Python playground to train an AI model to win at Arkanoid (NES)

![demo](demo.gif)

```
$ pip install -r requirements.txt
$ python3 main.py play --help
Usage: main.py play [OPTIONS]

Options:
  --render / --no-render          [default: render]
  --fps INTEGER                   [default: 50]
  --episodes INTEGER              [default: 3]
  --frames INTEGER
  --agent [heuristic|dqn|qlearning|human]
                                  [default: heuristic]
```

- Select how many episodes to go through with `--episodes` or give a total amount of frames with `--frames`.

- Pause at any time by pressing `P` and change to keyboard mode by pressing `H`

- Choose your player! `--agent`

    - `human`: Play with the keyboard
    - `heuristic`: Play with a deterministic non-AI agent that should be pretty good
    - `dqn`: Try the Deep Q Network agent
