import math
import operator as ops
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from agent import ArkAgent

BATCH_SIZE: int = 512


@dataclass
class StateTransition:
    screen: npt.NDArray[np.uint8]
    info: Dict
    action: int
    reward: Union[int, float]
    next_screen: npt.NDArray[np.uint8]
    next_info: Dict


from typing import Optional


def info_to_array(info):
    df = pd.json_normalize(info)
    df = df[sorted(df.columns)]
    list_columns = [col for col in df.columns if isinstance(df.loc[0, col], list)]
    df = pd.concat(
        [
            df.drop(columns=list_columns),
            *[df[col].apply(pd.Series).add_prefix(f"{col}.") for col in list_columns],
        ],
        axis=1,
    )
    return df.iloc[0].values


class ReplayMemory:
    def __init__(self, capacity: int = 512 * 16, batch_size: int = BATCH_SIZE):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def can_sample(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        return len(self.memory) > batch_size

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

    def sample(self, batch_size: Optional[int] = None):
        batch_size = batch_size or self.batch_size
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):
    def __init__(self, n_observations: int, n_actions: int, seed: int = 117):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.l1 = torch.nn.Linear(n_observations, 128)  # TODO: unhardcode
        self.l2 = torch.nn.Linear(128, 128)
        self.l3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.l1(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = self.l3(x)
        return x

    @classmethod
    def prepare(cls, info, _screen):
        retval = info_to_array(info)
        return retval


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
        self.policy_net = DQN(info_to_array(env.info).shape[0], env.action_space.n).to(
            self.device
        )
        self.target_net = DQN(info_to_array(env.info).shape[0], env.action_space.n).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.lr = 1e-4
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.batch_size = batch_size
        self.history = defaultdict(list)

    def toggle_train(self):
        self.train = not self.train

    @property
    def is_training(self):
        return self.train

    def get_action(self, screen, info):
        if self.is_training:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.training_steps_done / self.eps_decay
            )
            self.training_steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    expected_rewards = self.policy_net(
                        torch.tensor(
                            self.policy_net.prepare(info, screen),
                            device=self.device,
                            dtype=torch.float32,
                        ).unsqueeze(0)
                    )
                    return expected_rewards.max(1)[1].item()
            return self.env.action_space.sample(
                mask=np.array([1, 0, 0, 1], dtype=np.int8)
            )
            """return torch.tensor(
                [
                    [
                        self.env.action_space.sample(
                            mask=np.array([1, 0, 0, 1], dtype=np.int8)
                        )
                    ]
                ],
                device=self.device,
                dtype=torch.long,
            )"""

        # print("not training")
        with torch.no_grad():
            return (
                self.policy_net(
                    torch.tensor(
                        self.policy_net.prepare(info, screen),
                        device=self.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                )
                .max(1)[1]
                .item()
            )

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
        )

        state_batch = torch.tensor(
            np.array([self.policy_net.prepare(st.info, st.screen) for st in batch]),
            device=self.device,
            dtype=torch.float32,
        )
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

        self.history[self.env.episode].append(loss.item())

    def update(self, screen, info, action, reward, done, next_screen, next_info):
        # Store the transition in memory
        self.memory.push(screen, info, action, reward, next_screen, next_info)

        # One step optimization on the policy network
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
