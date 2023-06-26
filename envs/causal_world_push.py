import gym
import gym.wrappers
import numpy as np
from causal_world.envs import CausalWorld
from causal_world.task_generators import generate_task, PushingTaskGenerator
import cv2
from gym.utils import seeding


class CausalWorldPush(gym.Env):
    def __init__(self, image_size=128, seed=0, activate_sparse_reward=False, variables_space='space_a'):
        self.variable_space = variables_space
        self.activate_sparse_reward = activate_sparse_reward
        self._seed = seed
        self.max_radius = 0.18
        self.np_random, seed = seeding.np_random(self._seed)
        self.image_size = image_size
        shape = (self.image_size, self.image_size, 6 * 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.env = None

        task = PushingTaskGenerator(
            variables_space=self.variable_space,
            activate_sparse_reward=self.activate_sparse_reward
        )
        env = CausalWorld(
            task=task,
            observation_mode='pixel',
            normalize_observations=False,
            seed=self._seed,
        )
        self._action_space = gym.spaces.Box(low=env.action_space.low, high=env.action_space.high,
                                            shape=env.action_space.shape, dtype=np.float32)

    def _sample_position(self):
        length = self.np_random.uniform(0, self.max_radius * self.max_radius)
        angle = np.pi * self.np_random.uniform(0, 2)
        x = np.sqrt(length) * np.cos(angle)
        y = np.sqrt(length) * np.sin(angle)

        return x, y

    @property
    def action_space(self):
        return self._action_space

    def _compose_observation(self, observation):
        observation = observation.transpose(1, 2, 0, 3)
        observation = observation.reshape(*observation.shape[:2], self.observation_space.shape[2])
        observation = cv2.resize(observation, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        assert observation.shape == self.observation_space.shape
        return observation

    def reset(self):
        if self.env is not None:
            self.env.close()
        tool_block_position = self._sample_position()
        goal_block_position = self._sample_position()
        task = PushingTaskGenerator(
            variables_space=self.variable_space,
            fractional_reward_weight=1,
            tool_block_position=np.array([tool_block_position[0], tool_block_position[1], 0.0325]),
            goal_block_position=np.array([goal_block_position[0], goal_block_position[1], 0.0325]),
            activate_sparse_reward=self.activate_sparse_reward
        )

        self.env = CausalWorld(
            task=task,
            observation_mode='pixel',
            normalize_observations=False,
            seed=self._seed,
        )

        observation = self.env.reset()
        observation = self._compose_observation(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = self._compose_observation(observation)
        return observation, reward, done, info


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     env = CausalWorldPush(seed=0)
#     while True:
#         obs = env.reset()
#         obs = obs[:, :, :3]
#         plt.imshow(obs)
#         plt.show()
#
#         done = False
#         i = 0
#         while not done:
#             i += 1
#             action = env.action_space.sample()
#             _, _, done, _ = env.step(action)
#
#         print(i)
