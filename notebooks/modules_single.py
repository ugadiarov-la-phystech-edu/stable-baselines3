import json
import math
import random
import sys
import os
import time
from collections import deque
from typing import Dict, Type, Any, Tuple, NamedTuple, Optional, Union, Generator, List

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, RecordEpisodeStatistics, NormalizeReward, \
    FrameStack
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def make_env(n_envs, env_id, kwargs=None):
    kwargs = kwargs or {}
    def _make():
        env = gym.make(env_id, **kwargs)
        env = ImageExtractorWrapper(env)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = FrameStack(env, num_stack=3)
        env = RecordEpisodeStatistics(env)

        return env

    vec_env = AsyncVectorEnv([_make for _ in range(n_envs)])
    vec_env = NormalizeReward(vec_env)

    return vec_env


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
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam, optimizer_kwargs: Dict[str, Any] = {},):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = dict(optimizer_kwargs)
        if self.optimizer_class == torch.optim.Adam:
            self.optimizer_kwargs["eps"] = self.optimizer_kwargs.get("eps", 1e-5)

        self.features_extractor = self.make_features_extractor()

        self.action_net = nn.Linear(self.features_extractor.features_dim, self.action_space.n)

        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def make_features_extractor(self):
        return Encoder(self.observation_space)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)

        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        actions = distribution.mode() if deterministic else distribution.sample()
        actions = actions.reshape((-1, *self.action_space.shape))

        return actions

    def extract_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.float() / 255.
        features = self.features_extractor(obs)

        return features

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)

        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        log_probs = distribution.log_prob(actions_taken)
        entropy = distribution.entropy()

        return log_probs, entropy

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
            buffer_size: Optional[int] = None,
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
        self.gamma = gamma
        self.full = None
        self.generator_ready = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None

    def reset(self) -> None:
        self.observations = [[] for _ in range(self.n_envs)]
        self.actions = [[] for _ in range(self.n_envs)]
        self.rewards = [[] for _ in range(self.n_envs)]
        self.returns = [None for _ in range(self.n_envs)]
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
            env_ids,
    ) -> None:
        for env_id in env_ids:
            self.observations[env_id].append(obs[env_id])
            self.actions[env_id].append(action[env_id])
            self.rewards[env_id].append(reward[env_id])

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)

    def _get_samples(
            self,
            batch_inds: np.ndarray,
    ) -> BufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
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
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None or math.isinf(batch_size):
            batch_size = buffer_size

        start_idx = 0
        while start_idx < buffer_size:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def compute_returns_and_advantage(self) -> None:
        for env_id in range(self.n_envs):
            actions = self.actions[env_id]
            rewards = self.rewards[env_id]
            assert len(rewards) == len(actions)

            returns = [None] * len(actions)
            reward_to_go = 0
            for step in reversed(range(len(actions))):
                reward_to_go = rewards[step] + self.gamma * reward_to_go
                returns[step] = reward_to_go

            self.returns[env_id] = returns


def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_algorithm(env_kwargs, algorithm_class, algorithm_kwargs, use_wandb, wandb_kwargs, total_timesteps, log_interval):
    env = make_env(**env_kwargs)
    model = algorithm_class(env, **algorithm_kwargs)
    if use_wandb:
        config = dict(env_kwargs)
        config.update(algorithm_kwargs)
        config.update(dict(algorithm_class=algorithm_class, total_timesteps=total_timesteps, log_interval=log_interval))
        config.update(wandb_kwargs.get('config', {}))
        wandb_kwargs['config'] = config
        wandb.init(**wandb_kwargs)

    model.learn(total_timesteps=total_timesteps, log_interval=log_interval)


class MonteCarloActorCritic:
    def __init__(
            self,
            env,
            learning_rate: float = 3e-4,
            batch_size: Optional[int] = 64,
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

        self.observation_space = env.single_observation_space
        self.action_space = env.single_action_space
        self.n_envs = env.num_envs
        self.env = env

        self.n_steps = None
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
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def _update_info_buffer(self, infos: Dict[str, Any]) -> None:
        assert self.ep_info_buffer is not None

        if 'final_info' not in infos:
            return

        for idx, info in enumerate(infos['final_info']):
            if info is not None:
                maybe_ep_info = info.get("episode")
                if maybe_ep_info is not None:
                    self.ep_info_buffer.extend([maybe_ep_info])

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
    ) -> bool:
        self.policy.set_training_mode(False)
        self._last_obs, _ = self.env.reset()
        assert self._last_obs is not None, "No previous observation was provided"

        buffer.reset()
        dones = np.full(shape=(env.num_envs,), fill_value=False)
        while not dones.all().item():
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs, device=self.device)
                actions = self.policy(obs_tensor)
            actions = actions.cpu().numpy()
            new_obs, rewards, terminated, truncated, infos = env.step(actions)
            assert not np.any(truncated).item(), 'Episode truncation must be off'

            self.num_timesteps += env.num_envs - np.count_nonzero(dones)

            new_dones = terminated & (~dones)
            self._update_info_buffer(infos)
            actions = actions.reshape(-1, 1)

            self._episode_num += new_dones.sum().item()

            buffer.add(
                self._last_obs,
                actions,
                rewards,
                env_ids=np.nonzero(~dones)[0],
            )

            self._last_obs = new_obs
            dones |= new_dones
            self._last_episode_starts = dones

        buffer.compute_returns_and_advantage()

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)

        entropy_losses = []
        entropies = []
        pg_losses = []
        losses = []
        grad_norms = []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                advantages = rollout_data.returns
                if self.use_critic:
                    advantages = rollout_data.advantages

                # Normalize advantage
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy gradient loss
                policy_loss = -(advantages * log_prob).mean()
                pg_losses.append(policy_loss.item())

                # Entropy loss
                entropy = torch.mean(entropy)
                entropies.append(entropy.item())
                entropy_loss = -self.ent_coef * entropy
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + entropy_loss
                losses.append(loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.policy.optimizer.step()

            self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", np.mean(entropy_losses)), ("train/entropy", np.mean(entropies)),
            ("train/policy_gradient_loss", np.mean(pg_losses)), ("train/loss", np.mean(losses)),
            ("train/grad_norm", np.mean(grad_norms)), ("train/n_updates", self._n_updates)
        ])

    def _dump_logs(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.log_dict["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.log_dict["rollout/ep_rew_mean"] = np.mean(
                np.concatenate([ep_info["r"] for ep_info in self.ep_info_buffer])).item()
            self.log_dict["rollout/ep_len_mean"] = np.mean(
                np.concatenate([ep_info["l"] for ep_info in self.ep_info_buffer])).item()

        self.log_dict["time/fps"] = fps
        self.log_dict["time/time_elapsed"] = int(time_elapsed)
        self.log_dict["time/total_timesteps"] = self.num_timesteps
        self.log_dict["global_step"] = self.num_timesteps
        self.log_dict["num_episodes"] = self._episode_num
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
            self._last_obs, _ = self.env.reset(seed=self.seed)
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        iteration = 0
        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, self.rollout_buffer,)

            if not continue_training:
                break

            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        return self


if __name__ == '__main__':
    config_file_path = create_vizdoom_config()
    run_algorithm(
        env_kwargs=dict(n_envs=1, env_id='VizdoomCorridor-v0', kwargs={'scenario_file': config_file_path}),
        algorithm_class=MonteCarloActorCritic,
        algorithm_kwargs=dict(learning_rate=2.5e-4, batch_size=math.inf, n_epochs=1, gamma=0.99,
                              normalize_advantage=False, ent_coef=0.001, use_critic=False, seed=0,
                              policy_kwargs={'optimizer_class': torch.optim.RMSprop, 'optimizer_kwargs': {'eps': 1e-5},},
                              stats_window_size=10,
                              ),
        use_wandb=False,
        wandb_kwargs=dict(project='Test project', group='montecarlo_actor-critic', monitor_gym=True,
                          name='montecarlo_actor-critic'),
        total_timesteps=100000,
        log_interval=2,
    )