from typing import Union, Callable, Optional, Dict, Any, Type

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperObsType

from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame, ClipRewardEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from vizdoom import gymnasium_wrapper


class ImageExtractorWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['screen']

    def observation(self, observation: ObsType) -> WrapperObsType:
        return observation['screen']


class VizDoomWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
        clip_reward: bool = True,
    ) -> None:
        env = ImageExtractorWrapper(env)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)


def make_vizdoom_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=VizDoomWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )
