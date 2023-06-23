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
from typing import List
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


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


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


class DQN_CNN(nn.Module):
    def __init__(self, env):
        super().__init__()
        n_actions = env.action_space.n
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(384, 32),
            nn.GELU(),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x /= 255.0
        return self.network(x)


class DQN_TAB(nn.Module):
    def __init__(self, env, seed: int = 117, hidden_dim: int = 256):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(env.arrayinfo_shape, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, env.action_space.n)
        )

    def forward(self, x):
        return self.network(x)


class DQN(nn.Module):
    def __init__(self, env, seed: int = 117, hidden_dim: int = 256):
        super().__init__()
        self.dqn_tab = DQN_TAB(env, seed, hidden_dim)
        self.dqn_cnn = DQN_CNN(env)
        dqn_size = 256
        cnn_size = 32
        self.__l1 = nn.Linear(cnn_size, 16)
        self.__l2 = nn.Linear(16, 4)

    def forward(self, screen, info):
        x_cnn = self.dqn_cnn(screen)
        x_tab = self.dqn_tab(info)
        y = torch.concat((x_cnn, x_tab), dim=1)
        y = x_cnn
        y = self.__l1(y)
        y = F.relu(y)
        y = self.__l2(y)
        return y


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
        self.policy_net = DQN(env).to(self.device)
        self.target_net = DQN(env).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.lr = 5e-4
        self.gamma = 0.7
        self.tau = 1e-3
        self.eps_start = 0.95
        self.eps_end = 0.1
        self.eps_decay = 50
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.batch_size = batch_size
        self.loss = defaultdict(list)
        self.rewards = defaultdict(list)
        self.actions = defaultdict(
            lambda: {i: 0 for i in range(self.env.action_space.n)}
        )
        self.scores = dict()

    @property
    def durations(self):
        return {k: len(v) - 1 for k, v in self.rewards.items()}

    def toggle_train(self):
        self.train = not self.train

    @property
    def is_training(self):
        return self.train

    ##@profile
    def get_action(self, screen, info):
        if not isinstance(screen, torch.Tensor):
            screen = torch.tensor(
                rgb2gray(self.env.crop_screen(screen)),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        if not isinstance(info, torch.Tensor):
            info = torch.tensor(self.env.info_to_array(info), device=self.device)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.training_steps_done / self.eps_decay
        )
        self.training_steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                q_values = self.policy_net(screen.unsqueeze(0), info.unsqueeze(0))
                action = q_values.argmax(1).item()
                self.actions[self.env.episode][action] += 1
                return action
            # max_q_value = torch.tensor(
            #    [q_values.argmax(1).item()], device=self.device, dtype=torch.long
            # )
            # return max_q_value.item()
        else:
            action = self.env.action_space.sample(
                mask=np.array([1, 1, 1, 1], dtype=np.int8)
            )
            self.actions[self.env.episode][action] += 1
            return action
            # return torch.tensor(
            #    self.env.action_space.sample(
            #        mask=np.array([1, 1, 1, 1], dtype=np.int8)
            #    ),
            #    device=self.device,
            #    dtype=torch.long,
            # ).item()

    ##@profile
    def optimize_model(self):
        if not self.memory.can_sample():
            return

        batch = self.memory.sample()

        non_final_mask = torch.tensor(
            [x.next_screen is not None for x in batch],
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_infos = (
            torch.stack([t.next_info for t in batch if t.next_info is not None])
            .to(self.device)
            .to(dtype=torch.float32)
        )

        non_final_next_screens = (
            torch.cat([t.next_screen for t in batch if t.next_screen is not None])
            .to(self.device)
            .type(torch.float32)
        ).unsqueeze(1)

        screen_batch = (
            torch.cat([t.screen for t in batch if t.screen is not None])
            .to(self.device)
            .type(torch.float32)
        ).unsqueeze(1)

        info_batch = (
            torch.stack([t.info for t in batch if t.info is not None])
            .to(self.device)
            .type(torch.float32)
        )

        action_batch = torch.tensor(
            [t.action for t in batch], device=self.device, dtype=torch.long
        )

        reward_batch = torch.tensor(
            [t.reward for t in batch], device=self.device, dtype=torch.float32
        )

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(screen_batch, info_batch)
        # print(f"{state_action_values.shape=} {action_batch.shape=}")
        # print(f"{state_action_values}")
        # print(f"{action_batch}")
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(0))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.float32
        )

        with torch.no_grad():
            vals = self.target_net(non_final_next_screens, non_final_next_infos)
            next_state_values[non_final_mask] = vals.max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # print(f"{state_action_values.squeeze(0).shape=} {expected_state_action_values.unsqueeze(1).shape=}")
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values.squeeze(0), expected_state_action_values)
        self.loss[self.env.episode].append(loss)
        # Optimize the model
        self.optimizer.zero_grad()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        # TODO: should this go after backpropagation?
        loss.backward()
        self.optimizer.step()

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
        if not isinstance(screen, torch.Tensor):
            screen = torch.tensor(
                rgb2gray(self.env.crop_screen(screen)),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        if not isinstance(next_screen, torch.Tensor):
            next_screen = torch.tensor(
                rgb2gray(self.env.crop_screen(next_screen)),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
        if not isinstance(info, torch.Tensor):
            info = torch.tensor(self.env.info_to_array(info), device=self.device)
        if not isinstance(next_info, torch.Tensor):
            next_info = torch.tensor(
                self.env.info_to_array(next_info), device=self.device
            )

        # Store the transition in memory
        self.memory.push(screen, info, action, reward, next_screen, next_info)
        self.rewards[self.env.episode].append(reward)

        if self.is_training:
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
        agent = cls(env, batch_size=batch_size, train=False)
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
        save_every: Optional[int] = None,
    ):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_dir = checkpoint_dir / "latest"
        latest_dir.mkdir(exist_ok=True)

        if resume:
            agent = cls.load(env, latest_dir, batch_size=batch_size)
        else:
            agent = cls(env, batch_size=batch_size)

        c = count()

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        for episode in tqdm(range(1, episodes + 1), desc="Episode: ", position=0):
            screen, info = env.reset()
            episode_score = 0

            screen = torch.tensor(
                rgb2gray(env.crop_screen(screen)),
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0)
            info = torch.tensor(env.info_to_array(info), device=agent.device)

            for frame in tqdm(c, desc="Frame: ", position=1):
                action = agent.get_action(screen, info)
                next_screen, reward, done, next_info = env.step(action)

                next_screen = torch.tensor(
                    rgb2gray(env.crop_screen(next_screen)),
                    dtype=torch.float32,
                    device=agent.device,
                ).unsqueeze(0)
                next_info = torch.tensor(
                    env.info_to_array(next_info), device=agent.device
                )

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
                    losses = agent.loss[env.episode]
                    agent.scores[env.episode] = episode_score
                    print(
                        f"Episode {episode}: final score={env.game['score']} total rewards={episode_score} mean loss = {torch.mean(torch.tensor(losses)):.4f}",
                        flush=True,
                    )
                    break

        torch.save(agent.policy_net.state_dict(), checkpoint_dir / "policy_net.pth")
        torch.save(agent.policy_net.state_dict(), latest_dir / "policy_net.pth")
