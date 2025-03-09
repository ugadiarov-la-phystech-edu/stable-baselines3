import json
import math
import os
import random
import sys
import time
from collections import deque
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch as th
import gymnasium as gym
import wandb
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance, safe_mean

SelfA2C = TypeVar("SelfA2C", bound="A2C")


def set_seed_everywhere(seed: int, using_cuda: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if using_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Buffer:
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        self.device = torch.device(device)
        self.n_envs = n_envs
        self.gamma = gamma
        self.observations = None
        self.actions = None
        self.rewards = None
        self.returns = None
        self.values = None
        self.log_probs = None
        self.episode_starts = None
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

    def size(self) -> int:
        return sum(len(env_actions) for env_actions in self.actions)

    @staticmethod
    def flatten(buffer: list, dtype=None) -> np.ndarray:
        data = []
        for env_data in buffer:
            data.extend(env_data)

        return np.stack(data).astype(dtype)

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        log_prob = log_prob.clone().cpu().numpy()
        value = value.clone().flatten().cpu().numpy()
        for env_id in range(self.n_envs):
            self.observations[env_id].append(obs[env_id])
            self.actions[env_id].append(action[env_id])
            self.rewards[env_id].append(reward[env_id])
            self.episode_starts[env_id].append(episode_start[env_id])
            self.values[env_id].append(value[env_id])
            self.log_probs[env_id].append(log_prob[env_id])

    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, device=self.device)

    def get(self, batch_size=None) -> RolloutBufferSamples:
        self.observations = self.flatten(self.observations)
        self.actions = self.flatten(self.actions)
        self.advantages = self.flatten(self.advantages, dtype=np.float32)
        self.returns = self.flatten(self.returns, dtype=np.float32)
        self.log_probs = self.flatten(self.log_probs, dtype=np.float32)
        self.values = self.flatten(self.values, dtype=np.float32)

        data = (
            self.observations,
            self.actions,
            self.values,
            self.log_probs,
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
        )

        return (RolloutBufferSamples(*tuple(map(self.to_torch, data))),)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.flatten().cpu().numpy()
        for env_id in range(self.n_envs):
            values = self.values[env_id]
            actions = self.actions[env_id]
            rewards = self.rewards[env_id]
            episode_starts = self.episode_starts[env_id]
            assert len(values) == len(actions)
            assert len(rewards) == len(actions)
            assert len(episode_starts) == len(actions)

            returns = [None] * len(rewards)
            advantages = [None] * len(rewards)
            episode_starts.append(dones[env_id])
            reward_to_go = last_values[env_id]
            for step in reversed(range(len(rewards))):
                is_next_step_terminal = episode_starts[step + 1]
                reward_to_go = 0 if is_next_step_terminal else reward_to_go
                reward_to_go = rewards[step] + self.gamma * reward_to_go
                returns[step] = reward_to_go
                advantages[step] = reward_to_go - values[step]

            self.returns[env_id] = returns
            self.advantages[env_id] = advantages


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
    def __init__(self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch=None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class=None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = {"eps": 1e-5},
            use_q_critic: Optional[bool] = True,
            use_half_precision: Optional[bool] = True,):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr_schedule
        self.use_q_critic = use_q_critic
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        self.action_net = nn.Linear(self.features_dim, self.action_space.n)
        self.q_value_net = nn.Linear(self.features_dim, self.action_space.n)

        self.optimizer = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)

    def make_features_extractor(self):
        return Encoder(self.observation_space)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        actions = distribution.mode() if deterministic else distribution.sample()
        result = {'actions': actions, 'log_probs': distribution.log_prob(actions)}
        if self.use_q_critic and isinstance(self.action_space, spaces.Discrete):
            critic_output = self.q_value_net(features)
            values = critic_output
            result['q_values'] = values
            result['values'] = th.bmm(values.unsqueeze(1), distribution.probs.unsqueeze(2)).squeeze(
                dim=(1, 2))

        return result

    def extract_features(self, obs: torch.Tensor):
        obs = obs.float() / 255.
        features = self.features_extractor(obs)

        return features

    def evaluate_actions(self, obs: torch.Tensor, actions_taken: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        distribution = Categorical(logits=action_logits)
        entropy = distribution.entropy()
        result = {'entropy': entropy}
        if self.use_q_critic:
            if isinstance(self.action_space, spaces.Discrete):
                critic_output = self.q_value_net(features)
                values = critic_output
                result['q_values'] = values
                result['probs'] = distribution.probs

        return result

    def predict(self, observation: np.ndarray,
            state=None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,) -> th.Tensor:
        device = None
        for param in self.parameters():
            device = param.device
            break

        observation = torch.as_tensor(observation, device=device)
        if self.observation_space.shape == observation.shape[1:][::-1]:
            observation = observation.movedim(-1, 1)

        actions = self.forward(observation, deterministic=deterministic)['actions']
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        return actions, None

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)['values']

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class MyMAC:
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        pg_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        n_epochs: int = 1,
        device: Union[th.device, str] = "cuda",
        _init_setup_model: bool = True,
        detach_q_values: Optional[bool] = True,
        eval_env=None,
        n_eval_episodes=None,
        eval_interval=None,
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
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_interval = eval_interval or math.inf
        self.next_eval_step = self.eval_interval

        self.n_steps = n_steps
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.pg_coef = pg_coef
        self.max_grad_norm = max_grad_norm
        self.log_dict = {}
        self.evaluation_history = {}
        self.checkpoint_path = None

        self.set_random_seed(self.seed)
        self.rollout_buffer = Buffer(
            buffer_size=None,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gae_lambda=1,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.learning_rate, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)

    def _update_info_buffer(self, infos, dones=None) -> None:
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
        self.observation_space.seed(seed)

    def collect_rollouts(
            self,
            env,
            buffer: Buffer,
    ):
        self.policy.set_training_mode(False)
        assert self._last_obs is not None, "No previous observation was provided"

        step = 0
        buffer.reset()
        while step < self.n_steps:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_obs, device=self.device)
                policy_result = self.policy(obs_tensor)
                actions, values, log_probs = policy_result['actions'], policy_result['values'], policy_result['log_probs']
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos, dones)
            self._episode_num += dones.sum().item()
            actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = torch.as_tensor(infos[idx]["terminal_observation"], device=self.device)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs.unsqueeze(0)).item()
                    rewards[idx] += self.gamma * terminal_value

            buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            if self.num_timesteps >= self.next_eval_step:
                self.evaluate()
                self.next_eval_step += self.eval_interval

            step += 1

        with torch.no_grad():
            values = self.policy.predict_values(torch.as_tensor(new_obs, device=self.device))

        buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    def train(self) -> None:
        self.policy.set_training_mode(True)

        rollout_data = self.rollout_buffer.get()[0]
        actions = rollout_data.actions.long().flatten()
        policy_output = self.policy.evaluate_actions(rollout_data.observations, actions)
        prob = policy_output['probs']
        q_values = policy_output['q_values']
        entropy = policy_output['entropy']

        # Policy gradient loss
        policy_loss = -torch.mean(torch.bmm(q_values.unsqueeze(1).detach(), prob.unsqueeze(2)))

        # Q-value loss
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(dim=1)
        value_loss = self.vf_coef * F.mse_loss(q_values_taken, rollout_data.returns)

        # Entropy loss
        entropy = torch.mean(entropy)
        entropy_loss = -self.ent_coef * entropy

        loss = policy_loss + entropy_loss + value_loss

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        self._n_updates += 1

        self.log_dict = dict([
            ("train/entropy_loss", entropy_loss.item()), ("train/policy_gradient_loss", policy_loss.item()),
            ("train/value_loss", value_loss.item()), ("train/loss", loss.item()),
            ("train/grad_norm", grad_norm.item()), ("train/n_updates", self._n_updates),
            ("train/entropy", entropy.item()),
        ])

    def _dump_logs(self, iteration: int) -> None:
        assert self.ep_info_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.log_dict["time/iterations"] = iteration
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.log_dict["rollout/ep_rew_mean"] = np.mean(
                np.asarray([ep_info["r"] for ep_info in self.ep_info_buffer])).item()
            self.log_dict["rollout/ep_len_mean"] = np.mean(
                np.asarray([ep_info["l"] for ep_info in self.ep_info_buffer])).item()

        self.log_dict["time/fps"] = fps
        self.log_dict["time/time_elapsed"] = int(time_elapsed)
        self.log_dict["global_step"] = self.num_timesteps
        self.log_dict["time/total_timesteps"] = self.num_timesteps
        self.log_dict["time/total_episodes"] = self._episode_num
        print(json.dumps(self.log_dict, sort_keys=True, indent=4), flush=True)
        if wandb.run is not None:
            wandb.log(self.log_dict)

    def evaluate(self):
        self.policy.set_training_mode(False)
        if self.eval_env is None or self.n_eval_episodes is None:
            return

        n_eval_episodes = self.n_eval_episodes
        eval_env = self.eval_env
        n_envs = eval_env.num_envs
        returns = []
        lengths = []
        episode_counts = np.zeros(n_envs, dtype=np.int32)
        episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=np.int32)
        observations, _ = eval_env.reset()
        while (episode_counts < episode_count_targets).any():
            with torch.no_grad():
                obs_tensor = torch.as_tensor(np.array(observations), device=self.device)
                actions = self.policy(obs_tensor)['actions']
            actions = actions.cpu().numpy()
            new_observations, rewards, terminateds, truncateds, infos = eval_env.step(actions)
            assert not truncateds.any().item(), 'Episode truncation must be off'

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    if terminateds[i]:
                        assert 'final_info' in infos, 'Evaluation environments must be wrapped into RecordEpisodeStatistics'
                        info = infos['final_info'][i]
                        returns.append(info['episode']['r'].item())
                        lengths.append(info['episode']['l'].item())
                        episode_counts[i] += 1

            observations = new_observations

        assert len(returns) == n_eval_episodes
        assert len(lengths) == n_eval_episodes

        self.evaluation_history[self.num_timesteps] = {'returns': returns, 'lengths': lengths}
        self.save_checkpoint()

        log_eval_dict = {
            'time/total_episodes': self.num_timesteps,
            'eval/return': f'{np.mean(returns)} +/- {np.std(returns, ddof=1) / np.sqrt(len(returns))}',
            'eval/length': f'{np.mean(lengths)} +/- {np.std(lengths, ddof=1) / np.sqrt(len(lengths))}',
        }

        print(json.dumps(log_eval_dict, sort_keys=True, indent=4))

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            print('Skip checkpoint saving')
            return

        checkpoint = {
            'policy': self.policy.state_dict(), 'optimizer': self.policy.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps, 'episode_num': self._episode_num,
            'ep_info_buffer': self.ep_info_buffer, 'evaluation_history': self.evaluation_history,
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.checkpoint_path, 'checkpoint.pt'))
        print(f'Save checkpoint to {self.checkpoint_path}')

    def learn(
            self,
            total_timesteps: int,
            log_interval: int = 1,
            checkpoint_path: str = None,
    ):
        self.checkpoint_path = checkpoint_path
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
            self.collect_rollouts(self.env, self.rollout_buffer)

            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()


class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "CustomPolicy": Policy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        n_epochs: int = 1,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.normalize_advantage = normalize_advantage
        self.n_epochs = n_epochs

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.rollout_buffer_class = Buffer
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

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        entropy_losses = []
        policy_losses = []
        value_losses = []
        losses = []
        grad_norms = []

        for _ in range(self.n_epochs):
            # This will only loop once (get all data in one go)
            for rollout_data in self.rollout_buffer.get(batch_size=None):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                policy_result = self.policy.evaluate_actions(rollout_data.observations, actions)
                values, log_prob, entropy = policy_result['values'], policy_result['log_probs'], policy_result['entropy']
                values = values.flatten()

                # Normalize advantage (not present in the original implementation)
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy gradient loss
                policy_loss = -(advantages * log_prob).mean()
                policy_losses.append(policy_loss.item())

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)
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

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/grad_norm", np.mean(grad_norms))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfA2C:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


class MAC(A2C):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        pg_coef: float = 1.0,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        n_epochs: int = 1,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        detach_q_values: Optional[bool] = True,
    ):
        self.detach_q_values = detach_q_values
        if not policy_kwargs.get('use_q_critic', False):
            raise ValueError(f'It is mandatory to use Q-critic in MAC')

        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            gamma,
            gae_lambda,
            ent_coef,
            vf_coef,
            max_grad_norm,
            rms_prop_eps,
            use_rms_prop,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            normalize_advantage,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            n_epochs,
            device,
            _init_setup_model,
        )

        self.pg_coef = pg_coef

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            policy_result = self.policy.evaluate_actions(rollout_data.observations, actions)
            q_values, probs, entropy = policy_result['q_values'], policy_result['probs'], policy_result['entropy']
            q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(dim=1)

            # Normalize advantage (not present in the original implementation)
            # advantages = rollout_data.advantages
            # if self.normalize_advantage:
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            if self.detach_q_values:
                q_values = q_values.detach()

            policy_loss = -th.mean(th.bmm(q_values.unsqueeze(1), probs.unsqueeze(2)))

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(q_values_taken, rollout_data.returns)

            # Entropy loss favor exploration
            entropy_loss = -th.mean(entropy)

            loss = self.pg_coef * policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
