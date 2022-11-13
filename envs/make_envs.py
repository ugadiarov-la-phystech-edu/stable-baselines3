from stable_baselines3.common.monitor import Monitor
from .moldynamics_env import env_fn
from .subproc_env import SubprocVecEnv
from .wrappers import RewardWrapper

def make_env(env_kwargs, reward_wrapper_kwargs, seed, rank):
    def _thunk():
        env = env_fn(**env_kwargs)
        env.seed(seed + rank)
        env = RewardWrapper(env=env, **reward_wrapper_kwargs)
        env = Monitor(env)
        return env
    return _thunk

def make_envs(env_kwargs, reward_wrapper_kwargs, seed, num_processes, device):
    envs = [make_env(env_kwargs, reward_wrapper_kwargs, seed, i) for i in range(num_processes)]
    envs = SubprocVecEnv(device, envs)
    return envs