import json
import math
import random
import sys
import os
import time
from collections import deque
from typing import Dict, Type, Any, Tuple, NamedTuple, Optional, Union, Generator, List, SupportsFloat

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, RecordEpisodeStatistics, NormalizeReward, \
    FrameStack, TransformReward
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers.normalize import RunningMeanStd
from torch import nn
from torch.distributions import Categorical
import vizdoom
from vizdoom import gymnasium_wrapper

try:
    import wandb
    WANDB_IS_AVAILABLE = True
except ImportError:
    WANDB_IS_AVAILABLE = False

os.environ["WANDB_API_KEY"] = "5236b5f5547c7c158973c69d6aa8eeacca31e502"


def create_vizdoom_config(verbose=False):
    deadly_corridor_doom_skill_1 = """
    # Lines starting with # are treated as comments (or with whitespaces+#).
    # It doesn't matter if you use capital letters or not.
    # It doesn't matter if you use underscore or camel notation for keys, e.g. episode_timeout is the same as episodeTimeout.
    """
    deadly_corridor_doom_skill_1 += f"\ndoom_scenario_path = {os.path.relpath(os.path.dirname(vizdoom.__file__))}/scenarios/deadly_corridor.wad\n"
    deadly_corridor_doom_skill_1 += """
    # Skill 5 is recommended for the scenario to be a challenge.
    doom_skill = 1

    # Rewards
    death_penalty = 100
    #living_reward = 0

    # Rendering options
    screen_resolution = RES_320X240
    screen_format = CRCGCB
    render_hud = true
    render_crosshair = false
    render_weapon = true
    render_decals = false
    render_particles = false
    window_visible = true

    episode_timeout = 2100

    # Available buttons
    available_buttons =
      {
        MOVE_LEFT
        MOVE_RIGHT
        ATTACK
        MOVE_FORWARD
        MOVE_BACKWARD
        TURN_LEFT
        TURN_RIGHT
      }

    # Game variables that will be in the state
    available_game_variables = { HEALTH }

    mode = PLAYER
    """

    config_file_name = f'deadly_corridor_doom-skill-1.cfg'
    with open(config_file_name, 'w') as file_obj:
        print(deadly_corridor_doom_skill_1, file=file_obj)

    if verbose:
        print(deadly_corridor_doom_skill_1)

    return os.path.realpath(config_file_name)


class ImageExtractorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['screen']

    def observation(self, observation: Dict) -> np.ndarray:
        return observation['screen']


def make_env():
    config_file_path = create_vizdoom_config()
    env = gym.make('VizdoomCorridor-v0', **{'scenario_file': config_file_path})
    env = ImageExtractorWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=3)
    env = RecordEpisodeStatistics(env)
    env = TransformReward(env, f=lambda reward: reward / 100.)

    return env


def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512,
    ) -> None:
        super().__init__()
        self.features_dim = features_dim
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ReinforcePolicy(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr: float,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.RMSprop,
                 optimizer_kwargs: Dict[str, Any] = {'eps': 1e-5},):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.features_extractor = self.make_features_extractor()
        self.action_net = nn.Linear(self.features_extractor.features_dim, self.action_space.n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def make_features_extractor(self):
        return Encoder(self.observation_space)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float() / 255.
        features = self.features_extractor(obs)

        return features

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)

        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        actions = distribution.mode() if deterministic else distribution.sample()
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        log_probs = distribution.log_prob(actions_taken)
        entropy = distribution.entropy()

        return features, log_probs, entropy

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class BufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor


class Buffer:
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            gamma: float = 0.99,
            device: Union[torch.device, str] = "cpu",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = self.observation_space.shape
        self.device = torch.device(device)
        self.gamma = gamma
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None

    def reset(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.returns = None

    def size(self) -> int:
        return len(self.actions)

    def add(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)

    def _get_samples(
            self,
            batch_inds: np.ndarray,
    ) -> BufferSamples:
        observations = self.to_torch(self.observations[batch_inds])
        actions = self.to_torch(self.actions[batch_inds])
        returns = self.to_torch(self.returns[batch_inds])

        return BufferSamples(observations=observations, actions=actions, returns=returns)

    def get(self) -> BufferSamples:
        buffer_size = self.size()
        indices = np.random.permutation(buffer_size)
        self.observations = np.asarray(self.observations)
        self.actions = np.asarray(self.actions)
        self.returns = np.asarray(self.returns, dtype=np.float32)

        return self._get_samples(indices)

    def compute_returns(self) -> None:
        returns = [None] * len(self.rewards)
        reward_to_go = 0
        for step in reversed(range(len(self.rewards))):
            reward_to_go = self.rewards[step] + self.gamma * reward_to_go
            returns[step] = reward_to_go

        self.returns = returns


class Reinforce:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float = 2.5e-4,
            gamma: float = 0.99,
            ent_coef: float = 0.001,
            max_grad_norm: float = 0.5,
            stats_window_size: int = 10,
            policy_class: Type[ReinforcePolicy] = ReinforcePolicy,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            buffer_class: Type[Buffer] = Buffer,
            buffer_kwargs: Optional[Dict[str, Any]] = None,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "cuda",
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        print(f"Using {self.device} device")

        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.num_timesteps = 0
        self._num_timesteps_at_start = 0
        self._episode_num = 0
        self.seed = seed
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self._last_obs = None
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self._n_updates = 0

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env

        self.n_steps = None
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.log_dict = {}

        self.set_random_seed(self.seed)
        self.buffer_class = buffer_class
        self.buffer_kwargs = {} if buffer_kwargs is None else buffer_kwargs
        self.buffer = self.buffer_class(
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            **self.buffer_kwargs,
        )
        self.policy_class = policy_class
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def _update_info_buffer(self, info: Dict[str, Any]) -> None:
        # Collect statistics
        assert self.ep_info_buffer is not None

        ep_info = info.get("episode")
        if ep_info is not None:
            self.ep_info_buffer.append(ep_info)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        set_seed_everywhere(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def collect_rollouts(
            self,
            env,
            buffer: Buffer,
    ):
        self.policy.set_training_mode(False)
        self._last_obs, _ = self.env.reset()
        self._last_obs = np.array(self._last_obs)
        assert self._last_obs is not None, "No previous observation was provided"

        buffer.reset()
        terminated = False
        while not terminated:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs, device=self.device)
                action = self.policy(obs_tensor.unsqueeze(0))
            action = action.item()
            new_obs, reward, terminated, truncated, info = env.step(action)
            assert not truncated, 'Episode truncation must be off as we want to use Monte Carlo estimation'

            self.num_timesteps += 1
            self._episode_num += int(terminated)
            buffer.add(
                self._last_obs,
                action,
                reward,
            )

            self._last_obs = np.array(new_obs)
            self._update_info_buffer(info)

        buffer.compute_returns()

    def train(self) -> None:
        self.policy.set_training_mode(True)

        rollout_data = self.buffer.get()
        actions = rollout_data.actions.long()
        features, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        advantages = rollout_data.returns

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()

        # Entropy loss
        entropy = torch.mean(entropy)
        entropy_loss = -self.ent_coef * entropy

        loss = policy_loss + entropy_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", entropy_loss.item()), ("train/entropy", entropy.item()),
            ("train/policy_gradient_loss", policy_loss.item()), ("train/loss", loss.item()),
            ("train/grad_norm", grad_norm.item()), ("train/n_updates", self._n_updates)
        ])

    def get_current_mean_return(self) -> float:
        if len(self.ep_info_buffer) == 0:
            return -math.inf

        return np.mean(np.concatenate([ep_info["r"] for ep_info in self.ep_info_buffer])).item()

    def _dump_logs(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.log_dict["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.log_dict["rollout/ep_rew_mean"] = self.get_current_mean_return()
            self.log_dict["rollout/ep_len_mean"] = np.mean(
                np.concatenate([ep_info["l"] for ep_info in self.ep_info_buffer])).item()

        self.log_dict["time/fps"] = fps
        self.log_dict["time/time_elapsed"] = int(time_elapsed)
        self.log_dict["global_step"] = self.num_timesteps
        self.log_dict["num_episodes"] = self._episode_num
        print(json.dumps(self.log_dict, sort_keys=True, indent=4))

        if WANDB_IS_AVAILABLE and wandb.run is not None:
            wandb.log(self.log_dict)

    def learn(
            self,
            max_timesteps: int,
            expected_return: float,
            log_interval: int = 1,
    ) -> None:
        assert self.env is not None

        self.start_time = time.time_ns()
        self.num_timesteps = 0
        self._episode_num = 0
        self._num_timesteps_at_start = 0

        if self._last_obs is None:
            self._last_obs, _ = self.env.reset(seed=self.seed)

        iteration = 0
        while True:
            if self.get_current_mean_return() >= expected_return:
                print('Решено!')
                break

            if self.num_timesteps >= max_timesteps:
                print(f'Задача не решена за {max_timesteps} шагов')

            self.collect_rollouts(self.env, self.buffer, )
            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()


def run_algorithm(algorithm_class, algorithm_kwargs, use_wandb, wandb_kwargs, max_timesteps, log_interval, expected_return):
    env = make_env()
    model = algorithm_class(env, **algorithm_kwargs)
    if use_wandb:
        config = {}
        config.update(algorithm_kwargs)
        config.update(dict(algorithm_class=algorithm_class, max_timesteps=max_timesteps, log_interval=log_interval, expected_return=expected_return))
        config.update(wandb_kwargs.get('config', {}))
        wandb_kwargs['config'] = config
        wandb.init(**wandb_kwargs)

    model.learn(max_timesteps=max_timesteps, log_interval=log_interval, expected_return=expected_return)


class AverageReturnBuffer(Buffer):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, gamma: float = 0.99,
                 device: Union[torch.device, str] = "cpu"):
        super().__init__(observation_space, action_space, gamma, device)
        self.n_returns = 0
        self.average_return = 0

    def get_average_return(self):
        return self.average_return

    def compute_returns(self) -> None:
        super().compute_returns()
        self.average_return = (self.average_return * self.n_returns + sum(self.returns)) / (self.n_returns + len(self.returns))
        self.n_returns += len(self.returns)


class AverageReturnReinforce(Reinforce):
    def __init__(self, env: gym.Env, learning_rate: float = 2.5e-4, gamma: float = 0.99, ent_coef: float = 0.001,
                 max_grad_norm: float = 0.5, stats_window_size: int = 10, policy_class: Type[ReinforcePolicy] = ReinforcePolicy,
                 policy_kwargs: Optional[Dict[str, Any]] = None, buffer_class: Type[AverageReturnBuffer] = AverageReturnBuffer,
                 buffer_kwargs: Optional[Dict[str, Any]] = None, seed: Optional[int] = None,
                 device: Union[torch.device, str] = "cuda"):
        super().__init__(env, learning_rate, gamma, ent_coef, max_grad_norm, stats_window_size, policy_class, policy_kwargs,
                         buffer_class, buffer_kwargs, seed, device)

    def train(self) -> None:
        self.policy.set_training_mode(True)

        rollout_data = self.buffer.get()
        actions = rollout_data.actions.long()
        features, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        baseline = self.buffer.get_average_return()
        advantages = rollout_data.returns - baseline

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()

        # Entropy loss
        entropy = torch.mean(entropy)
        entropy_loss = -self.ent_coef * entropy

        loss = policy_loss + entropy_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", entropy_loss.item()), ("train/entropy", entropy.item()),
            ("train/policy_gradient_loss", policy_loss.item()), ("train/loss", loss.item()),
            ("train/grad_norm", grad_norm.item()), ("train/n_updates", self._n_updates),
            ("train/baseline", baseline)
        ])


class ValuePolicy(ReinforcePolicy):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr: float,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.RMSprop,
                 optimizer_kwargs: Dict[str, Any] = {'eps': 1e-5}):
        super().__init__(observation_space, action_space, lr, optimizer_class, optimizer_kwargs)
        self.value_net = nn.Linear(self.features_extractor.features_dim, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, log_probs, entropy = super().evaluate_actions(obs, actions_taken)
        values = self.value_net(features)

        return features, log_probs, entropy, values


class ValueBaselineReinforce(Reinforce):
    def __init__(self, env: gym.Env, learning_rate: float = 2.5e-4, gamma: float = 0.99, ent_coef: float = 0.001,
                 max_grad_norm: float = 0.5, stats_window_size: int = 10, policy_class: Type[ValuePolicy] = ValuePolicy,
                 policy_kwargs: Optional[Dict[str, Any]] = None, buffer_class: Type[Buffer] = Buffer,
                 buffer_kwargs: Optional[Dict[str, Any]] = None, seed: Optional[int] = None,
                 device: Union[torch.device, str] = "cuda", vf_coef: float = 0.5,):
        super().__init__(env, learning_rate, gamma, ent_coef, max_grad_norm, stats_window_size, policy_class,
                         policy_kwargs, buffer_class, buffer_kwargs, seed, device)
        self.vf_coef = vf_coef

    def train(self) -> None:
        self.policy.set_training_mode(True)

        rollout_data = self.buffer.get()
        actions = rollout_data.actions.long()
        features, log_prob, entropy, values = self.policy.evaluate_actions(rollout_data.observations, actions)
        advantages = rollout_data.returns - values.detach()
        baseline = values.mean().item()

        # Policy gradient loss
        policy_loss = -(advantages * log_prob).mean()

        # Entropy loss
        entropy = torch.mean(entropy)
        entropy_loss = -self.ent_coef * entropy

        # Value loss
        value_loss = self.vf_coef * F.mse_loss(values, rollout_data.returns.reshape_as(values))
        loss = policy_loss + entropy_loss + value_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", entropy_loss.item()), ("train/entropy", entropy.item()),
            ("train/policy_gradient_loss", policy_loss.item()), ("train/loss", loss.item()),
            ("train/grad_norm", grad_norm.item()), ("train/n_updates", self._n_updates),
            ("train/baseline", baseline), ("train/value_loss", value_loss.item())
        ])


if __name__ == '__main__':
    # run_algorithm(
    #     algorithm_class=Reinforce,
    #     algorithm_kwargs=dict(seed=0, ),
    #     use_wandb=False,
    #     wandb_kwargs=dict(project='Test project', group='reinforce', monitor_gym=True, name='reinforce'),
    #     max_timesteps=50000,
    #     log_interval=2,
    #     expected_return=2000,
    # )
    # run_algorithm(
    #     algorithm_class=AverageReturnReinforce,
    #     algorithm_kwargs=dict(seed=0, ),
    #     use_wandb=False,
    #     wandb_kwargs=dict(project='Test project', group='reinforce', monitor_gym=True, name='reinforce'),
    #     max_timesteps=50000,
    #     log_interval=2,
    #     expected_return=2000,
    # )
    run_algorithm(
        algorithm_class=ValueBaselineReinforce,
        algorithm_kwargs=dict(seed=0, ),
        use_wandb=False,
        wandb_kwargs=dict(project='Test project', group='reinforce', monitor_gym=True, name='reinforce'),
        max_timesteps=50000,
        log_interval=2,
        expected_return=2000,
    )
