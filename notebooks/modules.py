import json
import math
import random
import sys
import time
from collections import deque
from typing import Dict, Type, Any, Tuple, NamedTuple, Optional, Union, Generator, List

import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, RecordEpisodeStatistics, NormalizeReward, \
    TransformObservation, TimeLimit
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

try:
    import wandb
    WANDB_IS_AVAILABLE = True
except ImportError:
    WANDB_IS_AVAILABLE = False


def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ImageExtractorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['screen']

    def observation(self, observation: Dict) -> np.ndarray:
        return observation['screen']


class UnsqueezeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observation_space = env.observation_space
        assert len(observation_space.shape) == 2
        self.observation_space = spaces.Box(
            low=np.expand_dims(observation_space.low, axis=0),
            high=np.expand_dims(observation_space.high, axis=0),
            dtype=observation_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.expand_dims(observation, axis=0)


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


class Policy(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr: float,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam, optimizer_kwargs: Dict[str, Any] = {},
                 share_features_extractor: bool = True, normalize_images: bool = True,):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.normalize_images = normalize_images
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = dict(optimizer_kwargs)
        if self.optimizer_class == torch.optim.Adam:
            self.optimizer_kwargs["eps"] = self.optimizer_kwargs.get("eps", 1e-5)

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.action_net = nn.Linear(self.pi_features_extractor.features_dim, self.action_space.n)
        self.value_net = nn.Linear(self.vf_features_extractor.features_dim, 1)

        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def make_features_extractor(self):
        return Encoder(self.observation_space)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_features, vf_features = self.extract_features(obs)

        action_logits = self.action_net(pi_features)
        distribution = Categorical(logits=action_logits)
        actions = distribution.mode() if deterministic else distribution.sample()
        actions = actions.reshape((-1, *self.action_space.shape))
        log_probs = distribution.log_prob(actions)

        values = self.value_net(vf_features)

        return actions, log_probs, values

    def extract_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.float() / 255.
        pi_features = self.pi_features_extractor(obs)
        if self.share_features_extractor:
            vf_features = pi_features
        else:
            vf_features = self.vf_features_extractor(obs)

        return pi_features, vf_features

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_features, vf_features = self.extract_features(obs)

        action_logits = self.action_net(pi_features)
        distribution = Categorical(logits=action_logits)
        log_probs = distribution.log_prob(actions_taken)
        entropy = distribution.entropy()

        values = self.value_net(vf_features)

        return log_probs, values, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        _, vf_features = self.extract_features(obs)
        values = self.value_net(vf_features)

        return values

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class BufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class Buffer:
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        buffer_size: Optional[int] = None,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        device: Union[torch.device, str] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = self.observation_space.shape
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.full = None
        self.generator_ready = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.values = None
        self.episode_starts = None
        self.log_probs = None
        self.advantages = None

    def reset(self) -> None:
        self.observations = [[] for _ in range(self.n_envs)]
        self.actions = [[] for _ in range(self.n_envs)]
        self.rewards = [[] for _ in range(self.n_envs)]
        self.returns = [None for _ in range(self.n_envs)]
        self.episode_starts = [[] for _ in range(self.n_envs)]
        self.values = [[] for _ in range(self.n_envs)]
        self.log_probs = [[] for _ in range(self.n_envs)]
        self.advantages = [None for _ in range(self.n_envs)]
        self.generator_ready = False
        self.full = False

    def size(self) -> int:
        return sum(len(env_actions) for env_actions in self.actions)

    @staticmethod
    def swap_and_flatten(buffer: list) -> np.ndarray:
        data = []
        for env_data in buffer:
            data.extend(env_data)

        return np.stack(data)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.array,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        env_ids,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        for env_id in env_ids:
            self.observations[env_id].append(obs[env_id])
            self.actions[env_id].append(action[env_id])
            self.rewards[env_id].append(reward[env_id])
            self.episode_starts[env_id].append(episode_start[env_id])
            self.values[env_id].append(value[env_id].flatten().clone().cpu().numpy())
            self.log_probs[env_id].append(log_prob[env_id].clone().cpu().numpy())

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)

    def _get_samples(
            self,
            batch_inds: np.ndarray,
    ) -> BufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return BufferSamples(*tuple(map(self.to_torch, data)))

    def get(self, batch_size: Optional[Union[int, float]] = None) -> Generator[BufferSamples, None, None]:
        buffer_size = self.size()
        indices = np.random.permutation(buffer_size)
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None or math.isinf(batch_size):
            batch_size = buffer_size

        start_idx = 0
        while start_idx < buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def compute_returns_and_advantage(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        for env_id in range(self.n_envs):
            values = self.values[env_id]
            actions = self.actions[env_id]
            rewards = self.rewards[env_id]
            episode_starts = self.episode_starts[env_id]
            assert len(values) == len(actions)
            assert len(rewards) == len(actions)
            assert len(episode_starts) == len(actions)

            last_value = last_values[env_id]
            done = dones[env_id]
            advantages = [None] * len(actions)
            returns = [None] * len(actions)
            last_gae_term = 0
            for step in reversed(range(len(actions))):
                if step == len(actions) - 1:
                    next_non_terminal = 1.0 - done
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - episode_starts[step + 1]
                    next_value = values[step + 1]

                delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
                last_gae_term = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_term
                advantages[step] = last_gae_term
                returns[step] = last_gae_term + values[step]

            self.advantages[env_id] = advantages
            self.returns[env_id] = returns


class ActorCriticAlgorithm:
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: Union[int, float] = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        stats_window_size: int = 100,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        use_critic: Optional[bool] = True,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cuda",
    ):
        self.policy_class = Policy
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
        self._last_episode_starts = None
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self._n_updates = 0

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.log_dict = {}

        self.use_critic = use_critic

        self.set_random_seed(self.seed)
        self.rollout_buffer = Buffer(
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def _update_info_buffer(self, infos: List[Dict[str, Any]]) -> None:
        assert self.ep_info_buffer is not None

        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        set_seed_everywhere(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def collect_rollouts(
        self,
        env,
        buffer: Buffer,
        n_rollout_steps: int,
    ) -> bool:
        self.policy.set_training_mode(False)
        self._last_obs = self.env.reset()
        assert self._last_obs is not None, "No previous observation was provided"

        buffer.reset()
        dones = np.full(shape=(env.num_envs,), fill_value=False)
        while not dones.all().item():
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = torch.as_tensor(self._last_obs, device=self.device)
                actions, log_probs, values = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            new_obs, rewards, this_step_dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs - np.count_nonzero(dones)

            new_dones = this_step_dones & (~dones)
            self._update_info_buffer(infos)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self._episode_num += new_dones.sum()
            for idx, done in enumerate(new_dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    assert False, 'Episode truncation must be off'

            buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                env_ids=np.nonzero(~dones)[0],
            )

            self._last_obs = new_obs
            dones |= this_step_dones
            self._last_episode_starts = dones

        buffer.compute_returns_and_advantage(
            last_values=np.zeros((self.n_envs,), dtype=np.float32),
            dones=np.ones((self.n_envs,), dtype=bool)
        )

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)

        entropy_losses = []
        pg_losses, value_losses = [], []
        losses = []
        grad_norms = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                log_prob, values, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.returns
                if self.use_critic:
                    advantages = rollout_data.advantages

                # Normalize advantage
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy gradient loss
                policy_loss = -(advantages * log_prob).mean()
                pg_losses.append(policy_loss.item())

                value_loss = torch.as_tensor(0, dtype=torch.float32, device=self.device)
                if self.use_critic:
                    value_loss = F.mse_loss(values, rollout_data.returns)

                value_losses.append(value_loss.item())

                entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                losses.append(loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.policy.optimizer.step()

            self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", np.mean(entropy_losses)), ("train/policy_gradient_loss", np.mean(pg_losses)),
            ("train/value_loss", np.mean(value_losses)), ("train/loss", np.mean(losses)),
            ("train/grad_norm", np.mean(grad_norms)),  ("train/n_updates", self._n_updates)
        ])

    def _dump_logs(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.log_dict["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.log_dict["rollout/ep_rew_mean"] = np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            self.log_dict["rollout/ep_len_mean"] = np.mean([ep_info["l"] for ep_info in self.ep_info_buffer])

        self.log_dict["time/fps"] = fps
        self.log_dict["time/time_elapsed"] = int(time_elapsed)
        self.log_dict["time/total_timesteps"] = self.num_timesteps
        self.log_dict["global_step"] = self.num_timesteps
        print(json.dumps(self.log_dict, sort_keys=True, indent=4))

        if WANDB_IS_AVAILABLE and wandb.run is not None:
            wandb.log(self.log_dict)

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
    ):
        assert self.env is not None

        self.start_time = time.time_ns()
        self.num_timesteps = 0
        self._episode_num = 0
        self._num_timesteps_at_start = 0

        if self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        iteration = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        return self


class PPOAlgorithm:
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: Union[int, float] = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        stats_window_size: int = 100,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        use_critic: Optional[bool] = True,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "cuda",
        clip_range: float = 0.2,
    ):
        self.policy_class = Policy
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
        self._last_episode_starts = None
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = deque(maxlen=self._stats_window_size)
        self._n_updates = 0

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.log_dict = {}

        self.use_critic = use_critic

        self.set_random_seed(self.seed)
        self.rollout_buffer = Buffer(
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        self.clip_range = clip_range

    def _update_info_buffer(self, infos: List[Dict[str, Any]]) -> None:
        assert self.ep_info_buffer is not None

        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        set_seed_everywhere(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    def collect_rollouts(
        self,
        env,
        buffer: Buffer,
        n_rollout_steps: int,
    ) -> bool:
        self.policy.set_training_mode(False)
        assert self._last_obs is not None, "No previous observation was provided"

        n_steps = 0
        buffer.reset()
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = torch.as_tensor(self._last_obs, device=self.device)
                actions, log_probs, values = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self._episode_num += dones.sum()
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = torch.as_tensor(infos[idx]["terminal_observation"], device=self.device)
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)
                    rewards[idx] += self.gamma * terminal_value

            buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                env_ids=list(range(self.n_envs)),
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(torch.as_tensor(new_obs, device=self.device))

        buffer.compute_returns_and_advantage(last_values=values.clone().cpu().numpy().flatten(), dones=dones)

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)

        entropy_losses = []
        pg_losses, value_losses = [], []
        losses = []
        clip_fractions = []
        grad_norms = []

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                log_prob, values, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.returns
                if self.use_critic:
                    advantages = rollout_data.advantages

                # Normalize advantage
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                value_loss = torch.as_tensor(0, dtype=torch.float32, device=self.device)
                if self.use_critic:
                    value_loss = F.mse_loss(values, rollout_data.returns)

                value_losses.append(value_loss.item())

                entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                losses.append(loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())

                self.policy.optimizer.step()

            self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", np.mean(entropy_losses)), ("train/policy_gradient_loss", np.mean(pg_losses)),
            ("train/value_loss", np.mean(value_losses)), ("train/loss", np.mean(losses)),
            ("train/grad_norm", np.mean(grad_norms)),  ("train/n_updates", self._n_updates),
            ("train/clip_fraction", np.mean(clip_fractions)), ("train/clip_range", self.clip_range),
        ])

    def _dump_logs(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.log_dict["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.log_dict["rollout/ep_rew_mean"] = np.mean([ep_info["r"] for ep_info in self.ep_info_buffer])
            self.log_dict["rollout/ep_len_mean"] = np.mean([ep_info["l"] for ep_info in self.ep_info_buffer])

        self.log_dict["time/fps"] = fps
        self.log_dict["time/time_elapsed"] = int(time_elapsed)
        self.log_dict["time/total_timesteps"] = self.num_timesteps
        self.log_dict["global_step"] = self.num_timesteps
        print(json.dumps(self.log_dict, sort_keys=True, indent=4))

        if WANDB_IS_AVAILABLE and wandb.run is not None:
            wandb.log(self.log_dict)

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
    ):
        assert self.env is not None

        self.start_time = time.time_ns()
        self.num_timesteps = 0
        self._episode_num = 0
        self._num_timesteps_at_start = 0

        if self._last_obs is None:
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        iteration = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        return self


def make_env():
    from vizdoom import gymnasium_wrapper
    env = gym.make('VizdoomMyWayHome-v0')
    env = ImageExtractorWrapper(env)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = UnsqueezeWrapper(env)
    env = RecordEpisodeStatistics(env)

    return env


if __name__ == '__main__':
    vec_env = AsyncVectorEnv([make_env, make_env,],)
    vec_env = NormalizeReward(vec_env)
    obs, info = vec_env.reset()
    while True:
        obs, rew, term, trunc, info = vec_env.step(vec_env.action_space.sample())
        if np.any(term).item() or np.any(trunc).item():
            print()
