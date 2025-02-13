import io
import math
import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Dict, Any, Type, Union, Tuple, NamedTuple, Optional, Generator, List, Iterable

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch as th
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from gymnasium import spaces

from stable_baselines3.common import utils
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, \
    is_image_space_channels_first
from stable_baselines3.common.save_util import recursive_getattr, save_to_zip_file
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from vizdoom import gymnasium_wrapper

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import EpisodeBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance, safe_mean, update_learning_rate, \
    check_for_correct_spaces, set_random_seed
from stable_baselines3.common.vec_env import VecEnv, is_vecenv_wrapped, VecTransposeImage, unwrap_vec_normalize, \
    VecNormalize
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor


class ImageExtractorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['screen']

    def observation(self, observation: Dict) -> np.ndarray:
        return observation['screen']


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class ImageTransposeWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[2], *shape[:2]), dtype=env.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.transpose((2, 0, 1))


def make_env():
    return ImageTransposeWrapper(ResizeWrapper(ImageExtractorWrapper(gym.make('VizdoomMyWayHome-v0'))))


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
                 optimizer_class: Type[torch.optim.Optimizer] = th.optim.Adam, optimizer_kwargs: Dict[str, Any] = {},
                 features_extractor_class: Type[nn.Module] = Encoder, features_extractor_kwargs: Dict[str, Any] = {},
                 share_features_extractor: bool = True, normalize_images: bool = True,):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = dict(features_extractor_kwargs)
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
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

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
        self.action_dim = self.action_space.n
        self.pos = 0
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.pos = None
        self.full = None
        self.generator_ready = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.values = None
        self.log_probs = None
        self.advantages = None

    def reset(self) -> None:
        self.observations = [[] for _ in range(self.n_envs)]
        self.actions = [[] for _ in range(self.n_envs)]
        self.rewards = [[] for _ in range(self.n_envs)]
        self.returns = [None for _ in range(self.n_envs)]
        self.values = [[] for _ in range(self.n_envs)]
        self.log_probs = [[] for _ in range(self.n_envs)]
        self.advantages = [None for _ in range(self.n_envs)]
        self.generator_ready = False
        self.pos = 0
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
        value: torch.Tensor,
        log_prob: torch.Tensor,
        env_ids,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param env_ids: which environments produced data
        """
        if len(log_prob.shape) == 0:
            # unsqueeze: (N,) -> (N, 1)
            log_prob = log_prob.reshape(-1, 1)

        action = action.reshape((self.n_envs, self.action_dim))

        for env_id in env_ids:
            self.observations[env_id].append(obs[env_id])
            self.actions[env_id].append(action[env_id])
            self.rewards[env_id].append(reward[env_id])
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
        # Prepare the data
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

        # Return everything, don't create minibatches
        if batch_size is None or math.isinf(batch_size):
            batch_size = buffer_size

        start_idx = 0
        while start_idx < buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def compute_returns_and_advantage(self,) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        """
        for env_id in range(self.n_envs):
            values = self.values[env_id]
            actions = self.actions[env_id]
            rewards = self.rewards[env_id]
            assert len(values) == len(actions)
            assert len(rewards) == len(actions)

            advantages = [None] * len(actions)
            returns = [None] * len(actions)
            last_gae_term = 0
            next_values = 0
            for step in reversed(range(len(actions))):
                delta = rewards[step] + self.gamma * next_values - values[step]
                last_gae_term = delta + self.gamma * self.gae_lambda * last_gae_term
                advantages[step] = last_gae_term
                returns[step] = last_gae_term + values[step]
                next_values = values[step]

            self.advantages[env_id] = advantages
            self.returns[env_id] = returns


class EpisodicPGPPO:
    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: Union[int, float] = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[EpisodeBuffer]] = EpisodeBuffer,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        use_critic: Optional[bool] = True,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cuda",
        _init_setup_model: bool = True,
    ):
        self.policy_class = Policy
        self.device = device if isinstance(device, th.device) else th.device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env: Optional[VecEnv] = None
        self._vec_normalize_env = None

        # Create and wrap the env if needed
        env = self._wrap_env(env, self.verbose, monitor_wrapper)
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
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if normalize_advantage and batch_size is not None:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if batch_size is not None and not math.isinf(self.n_steps):
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.use_critic = use_critic

        if _init_setup_model:
            self._setup_model()

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # if self.rollout_buffer_class is None:
        #     if isinstance(self.observation_space, spaces.Dict):
        #         self.rollout_buffer_class = DictRolloutBuffer
        #     else:
        #         self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = self.learning_rate

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: EpisodeBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self._last_obs = self.env.reset()
        assert self._last_obs is not None, "No previous observation was provided"

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        dones = np.full(shape=(env.num_envs,), fill_value=False)
        while not dones.all().item():
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, log_probs, values = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, this_step_dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs - np.count_nonzero(dones)

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            new_dones = this_step_dones & (~dones)
            self._update_info_buffer(infos, new_dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(new_dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    assert False, 'Cannot have TimeLimit event in EpisodicPPO'
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                values,
                log_probs,
                env_ids=np.nonzero(~dones)[0],
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            dones |= this_step_dones

        rollout_buffer.compute_returns_and_advantage()

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        losses = []
        grad_norms = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                log_prob, values, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.returns
                if self.use_critic:
                    advantages = rollout_data.advantages

                # Normalize advantage
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy gradient loss
                policy_loss = -(advantages * log_prob).mean()
                pg_losses.append(policy_loss.item())

                value_loss = th.as_tensor(0, dtype=th.float32, device=self.device)
                if self.use_critic:
                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                losses.append(loss.item())

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/grad_norm", np.mean(grad_norms))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self):
        state_dicts = ["policy", "policy.optimizer"]
        if hasattr(self.policy, "scaler"):
            state_dicts.append("policy.scaler")

        return state_dicts, []


class MyPPO:
    def __init__(
        self,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: Union[int, float] = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = RolloutBuffer,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        use_critic: Optional[bool] = True,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cuda",
        _init_setup_model: bool = True,
    ):
        self.policy_class = Policy
        self.device = device if isinstance(device, th.device) else th.device(device)
        if verbose >= 1:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.action_noise = None
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._episode_num = 0
        # Used for gSDE only
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self._stats_window_size = stats_window_size
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom logger or not
        self._custom_logger = False
        self.env: Optional[VecEnv] = None
        self._vec_normalize_env = None

        # Create and wrap the env if needed
        env = self._wrap_env(env, self.verbose, monitor_wrapper)
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
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        if normalize_advantage and batch_size is not None:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if batch_size is not None and not math.isinf(self.n_steps):
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.use_critic = use_critic

        if _init_setup_model:
            self._setup_model()

    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        check_for_nested_spaces(env.observation_space)

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # if self.rollout_buffer_class is None:
        #     if isinstance(self.observation_space, spaces.Dict):
        #         self.rollout_buffer_class = DictRolloutBuffer
        #     else:
        #         self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = self.learning_rate

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def get_env(self) -> Optional[VecEnv]:
        """
        Returns the current environment (can be None if not defined).

        :return: The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env: GymEnv, force_reset: bool = True) -> None:
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        Furthermore wrap any non vectorized env into a vectorized
        checked parameters:
        - observation_space
        - action_space

        :param env: The environment for learning a policy
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        """
        # if it is not a VecEnv, make it a VecEnv
        # and do other transformations (dict obs, image transpose) if needed
        env = self._wrap_env(env, self.verbose)
        assert env.num_envs == self.n_envs, (
            "The number of environments to be set is different from the number of environments in the model: "
            f"({env.num_envs} != {self.n_envs}), whereas `set_env` requires them to be the same. To load a model with "
            f"a different number of environments, you must use `{self.__class__.__name__}.load(path, env)` instead"
        )
        # Check that the observation spaces match
        check_for_correct_spaces(env, self.observation_space, self.action_space)
        # Update VecNormalize object
        # otherwise the wrong env may be used, see https://github.com/DLR-RM/stable-baselines3/issues/637
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Discard `_last_obs`, this will force the env to reset before training
        # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
        if force_reset:
            self._last_obs = None

        self.n_envs = env.num_envs
        self.env = env

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        # self.env is always a VecEnv
        if self.env is not None:
            self.env.seed(seed)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, log_probs, values = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        grad_norms = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                log_prob, values, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                grad_norms.append(grad_norm.item())

                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/grad_norm", np.mean(grad_norms))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self):
        state_dicts = ["policy", "policy.optimizer"]
        if hasattr(self.policy, "scaler"):
            state_dicts.append("policy.scaler")

        return state_dicts, []

    def get_parameters(self) -> Dict[str, Dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


if __name__ == '__main__':
    env = AsyncVectorEnv([make_env, make_env, make_env],)
    o, i = env.reset()
    n = 0
    while True:
        o, r, term, trunc, i = env.step(np.asarray(env.action_space.sample()))
        n += 1
        print(n)
        if np.any(term).item() or np.any(trunc).item():
            print()