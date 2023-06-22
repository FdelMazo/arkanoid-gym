import math
import operator as ops
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from tqdm import tqdm
from itertools import count
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent import ArkAgent

BATCH_SIZE: int = 64


@dataclass
class StateTransition:
    screen: npt.NDArray[np.uint8]
    info: Dict
    action: int
    reward: Union[int, float]
    next_screen: npt.NDArray[np.uint8]
    next_info: Dict


# @profile
def info_to_array(info):
    return np.hstack(
        (
            pd.json_normalize(info)
            .drop(columns=sorted(["bricks.bricks_row"]))
            .iloc[0]
            .values,
            np.array(info["bricks"]["bricks_row"]).flatten(),
        )
    )


class ReplayMemory:
    def __init__(self, capacity: int = 512 * 16, batch_size: int = BATCH_SIZE):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    # @profile
    def can_sample(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        return len(self.memory) > batch_size

    # @profile
    def push(
        self,
        screen: npt.NDArray[np.uint8],
        info: Dict,
        action: int,
        reward: Union[int, float],
        next_screen: npt.NDArray[np.uint8],
        next_info: Dict,
    ):  # TODO: do args
        self.memory.append(
            StateTransition(screen, info, action, reward, next_screen, next_info)
        )

    # @profile
    def sample(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        env,
        seed: int = 117,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # 3 x 240 x 256
        self.__conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4, bias=False)
        # 32 x 59 x 63
        self.__conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        # 64 x 28 x 30
        self.__conv3 = torch.nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)
        # 64 x 14 x 15
        self.__fc1 = torch.nn.Linear(13440, 1024)
        self.__fc2 = torch.nn.Linear(1024, n_actions)
        # self.l1 = torch.nn.Linear(n_observations, hidden_dim)
        # self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.l3 = torch.nn.Linear(hidden_dim, n_actions)

    # @profile
    def forward(self, x):
        x /= 255
        for l in [self.__conv1, self.__conv2, self.__conv3]:
            x = torch.nn.functional.relu(l(x))
        x = x.flatten()
        x = torch.nn.functional.relu(self.__fc1(x))
        x = self.__fc2(x)
        # x = torch.nn.functional.relu(self.l1(x))
        # x = torch.nn.functional.relu(self.l2(x))
        # x = self.l3(x)
        return x


class DQNAgent(ArkAgent):
    def __init__(
        self,
        env,
        device=None,
        train: bool = True,
        training_eps: int = 50,
        batch_size: int = BATCH_SIZE,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.memory = ReplayMemory(batch_size * 32, batch_size)
        self.env = env
        self.train = train
        self.training_eps = training_eps
        self.training_eps_done = 0
        self.training_steps_done = 0
        self.episode_durations = []
        self.policy_net = DQN(
            info_to_array(env.info).shape[0], env.action_space.n, env
        ).to(self.device)
        self.target_net = DQN(
            info_to_array(env.info).shape[0], env.action_space.n, env
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.lr = 5e-3
        self.gamma = 0.3
        self.tau = 1e-3
        self.eps_start = 0.95
        self.eps_end = 0.05
        self.eps_decay = 20
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.batch_size = batch_size
        self.loss = defaultdict(list)
        self.rewards = defaultdict(list)

    def toggle_train(self):
        self.train = not self.train

    @property
    def is_training(self):
        return self.train

    ##@profile
    def get_action(self, screen, info):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.training_steps_done / self.eps_decay
        )
        self.training_steps_done += 1

        if sample > eps_threshold:
            state = (
                torch.tensor(
                    self.policy_net.prepare(info, screen).copy(),
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            self.policy_net.eval()
            with torch.no_grad():
                expected_rewards = self.policy_net(state)
            self.policy_net.train()
            return expected_rewards.max(0)[1].item()
        else:
            return self.env.action_space.sample(
                mask=np.array([1, 1, 1, 1], dtype=np.int8)
            )

    ##@profile
    def optimize_model(self):
        if not self.memory.can_sample():
            return

        batch = self.memory.sample()
        non_final_mask = torch.tensor(
            [st.next_info is not None for st in batch],
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.tensor(
            np.array(
                [
                    self.target_net.prepare(st.next_info, st.next_screen)
                    for st in batch
                    if st.next_info is not None and st.next_screen is not None
                ]
            ),
            device=self.device,
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)

        state_batch = torch.tensor(
            np.array([self.policy_net.prepare(st.info, st.screen) for st in batch]),
            device=self.device,
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)

        action_batch = torch.tensor(
            [(st.action,) for st in batch], device=self.device, dtype=torch.int64
        )
        reward_batch = torch.tensor(
            [st.reward for st in batch], device=self.device, dtype=torch.float32
        )

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(self.policy_net(state_batch).shape, action_batch.shape)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states
            ).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.loss[self.env.episode].append(loss.item())

    def soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    # @profile
    def update(self, screen, info, action, reward, done, next_screen, next_info):
        # Store the transition in memory
        self.memory.push(screen, info, action, reward, next_screen, next_info)
        self.rewards[self.env.episode].append(reward)
        # One step optimization on the policy network
        self.optimize_model()
        self.soft_update()

    def plot_history(self, last: int = 10):
        fig, (ax_loss, ax_rew) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

        start = 0
        for ep, series in deque(self.loss.items(), last):
            ax_loss.plot(np.arange(start, start + len(series)), series, label=f"{ep}")
            start += len(series)
        ax_loss.set_title("Loss")

        for ep, series in deque(self.rewards.items(), last):
            ax_rew.plot(series, label=f"{ep}")
        ax_rew.set_title("Rewards")

        plt.legend()
        plt.show()

    @classmethod
    def load(cls, env, checkpoint_dir: pathlib.Path, batch_size: int = 128):
        agent = cls(env, batch_size=batch_size)
        agent.policy_net.load_state_dict(torch.load(checkpoint_dir / "policy_net.pth"))
        agent.policy_net.eval()
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.target_net.eval()
        return agent

    @classmethod
    def train(
        cls,
        env,
        checkpoint_dir: pathlib.Path,
        batch_size: int = 128,
        episodes: int = 1000,
        resume: bool = False,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        save_every: Optional[int] = None,
    ):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_dir = checkpoint_dir / "latest"
        latest_dir.mkdir(exist_ok=True)

        if resume:
            agent = cls.load(env, latest_dir, batch_size=batch_size)
        else:
            agent = cls(env, batch_size=batch_size)

        screen, info = env.reset()

        c = count()

        for episode in tqdm(range(1, episodes + 1), desc="Episode: ", position=0):
            episode_score = 0

            for frame in tqdm(c, desc="Frame: ", position=1):
                action = agent.get_action(screen, info)
                next_screen, reward, done, next_info = env.step(action)
                episode_score += reward
                agent.update(screen, info, action, reward, done, next_screen, next_info)

                screen = next_screen
                info = next_info

                if save_every is not None and frame % save_every == 0:
                    checkpoint_dir_ = checkpoint_dir / f"{frame}"
                    checkpoint_dir_.mkdir(exist_ok=True)
                    torch.save(
                        agent.policy_net.state_dict(),
                        checkpoint_dir_ / "policy_net.pth",
                    )
                    torch.save(
                        agent.policy_net.state_dict(),
                        latest_dir / "policy_net.pth",
                    )

                if done:
                    losses = agent.loss[max(agent.loss.keys())]
                    print(
                        f"Episode {episode}: final score={env.game['score']} total rewards={episode_score} mean loss = {np.mean(losses):.4f}",
                        flush=True,
                    )
                    screen, info = env.reset()
                    break

        torch.save(agent.policy_net.state_dict(), checkpoint_dir / "policy_net.pth")
        torch.save(agent.policy_net.state_dict(), latest_dir / "policy_net.pth")
