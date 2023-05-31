import random

import gym
import numpy as np
import skimage
from gym.utils import seeding
from gym import spaces
import sys
import copy

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [c0 + width // 2, c0, c0 + width // 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    if width == 5:
        rr = np.asarray([r0 + 4] * 5 + [r0 + 3] * 3 + [r0 + 2] * 3 + [r0 + 1] + [r0], dtype=np.int32)
        cc = np.asarray(list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) * 2 + [c0 + 2] * 2, dtype=np.int32)
        return rr, cc

    rr, cc = [r0, r0 + width - 1, r0 + width - 1], [c0 + width // 2, c0, c0 + width - 1]
    return skimage.draw.polygon(rr, cc, im_size)


def circle(r0, c0, width, im_size):
    if width == 5:
        rr = np.asarray([r0] + [r0 + 1] * 3 + [r0 + 2] * 5 + [r0 + 3] * 3 + [r0 + 4], dtype=np.int32)
        cc = np.asarray(
            [c0 + 2] + list(range(c0 + 1, c0 + 4)) + list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) + [c0 + 2],
            dtype=np.int32)
        return rr, cc

    radius = width // 2
    return skimage.draw.ellipse(
        r0 + radius, c0 + radius, radius, radius)


def cross(r0, c0, width, im_size):
    diff1 = width // 3 + 1
    diff2 = 2 * width // 3
    rr = [r0 + diff1, r0 + diff2, r0 + diff2, r0 + width, r0 + width,
          r0 + diff2, r0 + diff2, r0 + diff1, r0 + diff1, r0, r0, r0 + diff1]
    cc = [c0, c0, c0 + diff1, c0 + diff1, c0 + diff2, c0 + diff2, c0 + width,
          c0 + width, c0 + diff2, c0 + diff2, c0 + diff1, c0 + diff1]
    return skimage.draw.polygon(rr, cc, im_size)


def pentagon(r0, c0, width, im_size):
    diff1 = width // 3 - 1
    diff2 = 2 * width // 3 + 1
    rr = [r0 + width // 2, r0 + width, r0 + width, r0 + width // 2, r0]
    cc = [c0, c0 + diff1, c0 + diff2, c0 + width, c0 + width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def parallelogram(r0, c0, width, im_size):
    if width == 5:
        rr = np.asarray([r0] * 2 + [r0 + 1] * 3 + [r0 + 2] * 3 + [r0 + 3] * 3 + [r0 + 4] * 2, dtype=np.int32)
        cc = np.asarray(
            [c0, c0 + 1] + list(range(c0, c0 + 3)) + list(range(c0 + 1, c0 + 4)) + list(range(c0 + 2, c0 + 5)) + list(
                range(c0 + 3, c0 + 5)), dtype=np.int32)
        return rr, cc

    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width // 2], [c0 + width - width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


class Push(gym.Env):
    GOAL = 'goal'
    PASSIVE_BOX = 'passive_box'
    ACTIVE_BOX = 'active_box'
    MOVING_BOXES_KEY = 'moving_boxes'

    STEP_REWARD = -0.01
    OUT_OF_FIELD_REWARD = -0.1
    COLLISION_REWARD = -0.1
    DEATH_REWARD = -1
    HIT_GOAL_REWARD = 1
    DESTROY_GOAL_REWARD = -1

    def __init__(self, n_active_boxes=4, n_passive_boxes=0, n_goals=1, width=5,
                 return_state=True, observation_type='shapes', max_episode_steps=75,
                 border_walls=True, channels_first=True, channel_wise=False, seed=None, render_scale=10,
                 ternary_interactions=False, do_reward_push_only=False, do_reward_active_box=True,
                 ):
        self.w = width
        self.step_limit = max_episode_steps

        assert n_active_boxes > 0
        self.n_active_boxes = n_active_boxes
        self.n_passive_boxes = n_passive_boxes

        assert n_goals > 0
        self.n_goals = n_goals
        self.ternary_interactions = ternary_interactions
        self.do_reward_push_only = do_reward_push_only
        self.do_reward_active_box = do_reward_active_box

        assert self.do_reward_active_box or self.n_passive_boxes > 0

        self.active_box_ids = set(range(self.n_active_boxes))
        self.passive_box_ids = set(range(self.n_active_boxes, self.n_active_boxes + self.n_passive_boxes))
        self.goal_ids = set(range(
            self.n_active_boxes + self.n_passive_boxes, self.n_active_boxes + self.n_passive_boxes + self.n_goals
        ))

        assert len(self.active_box_ids) == self.n_active_boxes
        assert len(self.passive_box_ids) == self.n_passive_boxes
        assert len(self.goal_ids) == self.n_goals

        self.render_scale = render_scale
        self.border_walls = border_walls
        self.colors = get_colors(num_colors=max(9, self._get_n_boxes()))
        self.observation_type = observation_type
        self.return_state = return_state
        self.channels_first = channels_first
        self.channel_wise = channel_wise

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }
        self.direction2action = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}

        self.np_random = None
        self.action_space = spaces.Discrete(4 * self.n_active_boxes)

        if self.observation_type == 'grid':
            raise NotImplementedError(f'Observation type "{observation_type}" has not been implemented yet')
            # channels are movable boxes, goals, static boxes
            self.observation_space = spaces.Box(
                0,
                1,
                (self.w, self.w, self.n_boxes)
            )
        elif self.observation_type in ('squares', 'shapes'):
            observation_shape = (self.w * self.render_scale, self.w * self.render_scale, self._get_n_channels())
            if self.channels_first:
                observation_shape = (observation_shape[2], *observation_shape[:2])
            self.observation_space = spaces.Box(0, 255, observation_shape, dtype=np.uint8)
        else:
            raise ValueError(f'Invalid observation_type: {self.observation_type}.')

        self.state = None
        self.steps_taken = 0
        self.box_pos = np.zeros(shape=(self._get_n_boxes(), 2), dtype=np.int32)
        self.speed = [{direction: 1 for direction in self.direction2action} for _ in
                      range(self.n_active_boxes + self.n_passive_boxes)]
        self.hit_goal_reward = [Push.HIT_GOAL_REWARD] * (self.n_active_boxes + self.n_passive_boxes)
        self.active_box_ids_in_game = None
        self.passive_box_ids_in_game = None

        self.seed(seed)
        self.reset()

    def _get_n_boxes(self):
        return self.n_active_boxes + self.n_passive_boxes + self.n_goals

    def _get_n_channels(self):
        if not self.channel_wise:
            return 3

        return self._get_n_boxes()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation(self):
        if self.observation_type == 'squares':
            image = self.render_squares()
        elif self.observation_type == 'shapes':
            image = self.render_shapes()
        else:
            assert False, f'Invalid observation type: {self.observation_type}.'

        if self.return_state:
            return np.array(self.state), image

        return image

    def _sample_coordinates(self, n, filed_width, occupied_coordinates=()):
        possible_locations = np.arange(filed_width ** 2)
        p = np.full_like(possible_locations, fill_value=1 / (filed_width ** 2 - len(occupied_coordinates)),
                         dtype=np.float32)
        for x, y in occupied_coordinates:
            p[x * filed_width + y] = 0

        locations = self.np_random.choice(possible_locations, n, replace=False, p=p)
        return np.unravel_index(locations, [filed_width, filed_width])

    def reset(self):
        self.state = np.full(shape=[self.w, self.w], fill_value=-1, dtype=np.int32)
        n_coordinates_in_central_area = self.n_passive_boxes
        n_coordinates_in_full_area = self.n_active_boxes
        if self.n_passive_boxes == 0:
            n_coordinates_in_full_area += self.n_goals
        else:
            n_coordinates_in_central_area += self.n_goals

        passive_box_xs, passive_box_ys = self._sample_coordinates(n_coordinates_in_central_area, self.w - 2)
        passive_box_xs += 1
        passive_box_ys += 1

        active_box_xs, active_box_ys = self._sample_coordinates(n_coordinates_in_full_area, self.w,
                                                                occupied_coordinates=list(zip(passive_box_xs,
                                                                                              passive_box_ys)))

        if self.n_passive_boxes == 0:
            goal_xs = active_box_xs[self.n_active_boxes:]
            goal_ys = active_box_ys[self.n_active_boxes:]
            active_box_xs = active_box_xs[:self.n_active_boxes]
            active_box_ys = active_box_ys[:self.n_active_boxes]
        else:
            goal_xs = passive_box_xs[self.n_passive_boxes:]
            goal_ys = passive_box_ys[self.n_passive_boxes:]
            passive_box_xs = passive_box_xs[:self.n_passive_boxes]
            passive_box_ys = passive_box_ys[:self.n_passive_boxes]

        for box_id, x, y in zip(self.passive_box_ids, passive_box_xs, passive_box_ys):
            assert self.state[x, y] == -1
            self.state[x, y] = box_id
            self.box_pos[box_id, :] = x, y

        for box_id, x, y in zip(self.active_box_ids, active_box_xs, active_box_ys):
            assert self.state[x, y] == -1
            self.state[x, y] = box_id
            self.box_pos[box_id, :] = x, y

        for box_id, x, y in zip(self.goal_ids, goal_xs, goal_ys):
            assert self.state[x, y] == -1
            self.state[x, y] = box_id
            self.box_pos[box_id, :] = x, y

        self.steps_taken = 0
        self.active_box_ids_in_game = self.active_box_ids.copy()
        self.passive_box_ids_in_game = self.passive_box_ids.copy()

        return self._get_observation()

    def _get_box_type(self, box_id):
        if box_id in self.active_box_ids:
            return Push.ACTIVE_BOX

        if box_id in self.passive_box_ids:
            return Push.PASSIVE_BOX

        if box_id in self.goal_ids:
            return Push.GOAL

        raise ValueError(f'Unexpected box_id={box_id}')

    def _destroy_box(self, box_id):
        box_pos = self.box_pos[box_id]
        self.state[box_pos[0], box_pos[1]] = -1
        self.box_pos[box_id] = -1, -1
        if box_id in self.active_box_ids_in_game:
            self.active_box_ids_in_game.remove(box_id)
        elif box_id in self.passive_box_ids_in_game:
            self.passive_box_ids_in_game.remove(box_id)

    def _move(self, box_id, new_pos):
        old_pos = self.box_pos[box_id]
        self.state[old_pos[0], old_pos[1]] = -1
        self.state[new_pos[0], new_pos[1]] = box_id
        self.box_pos[box_id] = new_pos

    def _is_free_cell(self, pos):
        return self.state[pos[0], pos[1]] == -1

    def _get_occupied_box_id(self, pos):
        return self.state[pos[0], pos[1]]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        box_id = action // len(self.directions)
        direction = tuple(self.directions[action % len(self.directions)].tolist())
        speed = self.speed[box_id][direction]
        reward = 0
        moving_boxes = [0] * self._get_n_boxes()

        for sub_step in range(speed):
            is_last_sub_step = sub_step == speed - 1
            observation, sub_reward, done, info = self.sub_step(action, return_observation=is_last_sub_step,
                                                                increment_step=is_last_sub_step)
            reward += sub_reward
            moving_boxes = [max(flag1, flag2) for flag1, flag2 in zip(moving_boxes, info[Push.MOVING_BOXES_KEY])]

        info[Push.MOVING_BOXES_KEY] = moving_boxes

        return observation, reward, done, info

    def sub_step(self, action, return_observation=False, increment_step=False):
        step_vector = self.directions[action % len(self.directions)]
        box_id = action // len(self.directions)
        assert box_id in self.active_box_ids

        box_old_pos = self.box_pos[box_id]
        box_new_pos = box_old_pos + step_vector

        reward = Push.STEP_REWARD if increment_step else 0
        moving_boxes = [0] * self._get_n_boxes()
        moving_boxes[box_id] = 1

        if not self._is_in_grid(box_old_pos, box_id):
            # This box is out of the game. There is nothing to do.
            pass
        elif not self._is_in_grid(box_new_pos, box_id):
            reward += Push.OUT_OF_FIELD_REWARD
            if not self.border_walls:
                self._destroy_box(box_id)
        elif not self._is_free_cell(box_new_pos):
            # push into another box
            another_box_id = self._get_occupied_box_id(box_new_pos)
            another_box_type = self._get_box_type(another_box_id)

            if another_box_type == Push.GOAL:
                if self.do_reward_push_only or not self.do_reward_active_box:
                    reward += Push.COLLISION_REWARD
                else:
                    reward += self.hit_goal_reward[box_id]
                    self._destroy_box(box_id)
            else:
                if self.ternary_interactions:
                    moving_boxes[another_box_id] = 1
                    another_box_new_pos = box_new_pos + step_vector
                    if self._is_in_grid(another_box_new_pos, another_box_id):
                        if self._is_free_cell(another_box_new_pos):
                            self._move(another_box_id, another_box_new_pos)
                            self._move(box_id, box_new_pos)
                        elif self._get_box_type(self._get_occupied_box_id(another_box_new_pos)) == Push.GOAL:
                            if another_box_type == Push.PASSIVE_BOX or self.do_reward_active_box:
                                reward += self.hit_goal_reward[another_box_id]
                                self._destroy_box(another_box_id)
                                self._move(box_id, box_new_pos)
                            else:
                                reward += Push.COLLISION_REWARD
                        else:
                            reward += Push.COLLISION_REWARD
                    else:
                        reward += Push.OUT_OF_FIELD_REWARD
                        if not self.border_walls:
                            self._destroy_box(another_box_id)
                            self._move(box_id, box_new_pos)
                else:
                    reward += Push.COLLISION_REWARD
        else:
            # pushed into open space, move box
            self._move(box_id, box_new_pos)

        if increment_step:
            self.steps_taken += 1

        info = {Push.MOVING_BOXES_KEY: moving_boxes}

        if not self.do_reward_active_box:
            done = len(self.passive_box_ids_in_game) == 0
        elif self.do_reward_push_only:
            done = len(self.passive_box_ids_in_game) == 0 and len(self.active_box_ids_in_game) == 1
        else:
            done = len(self.active_box_ids_in_game) == 0

        if not done and self.steps_taken >= self.step_limit:
            info["TimeLimit.truncated"] = True
            done = True

        observation = self._get_observation() if return_observation else None
        return observation, reward, done, info

    def _is_in_grid(self, point, box_id):
        if self._get_box_type(box_id) == Push.PASSIVE_BOX:
            return (1 <= point[0] < self.w - 1) and (1 <= point[1] < self.w - 1)

        return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

    def print(self, message=''):
        state = self.state
        chars = {-1: '.'}
        for box_id in range(self._get_n_boxes()):
            box_type = self._get_box_type(box_id)
            if box_type == Push.GOAL:
                chars[box_id] = 'x'
            elif box_type in (Push.ACTIVE_BOX, Push.PASSIVE_BOX):
                chars[box_id] = '#'
            else:
                chars[box_id] = '@'

        pretty = "\n".join(["".join([chars[x] for x in row]) for row in state])
        print(pretty)
        print("TIMELEFT ", self.step_limit - self.steps_taken, message)

    def clone_full_state(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def restore_full_state(self, state_dict):
        self.__dict__.update(state_dict)

    def get_action_meanings(self):
        return ["down", "left", "up", "right"] * len(self.active_box_ids)

    def render_squares(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_n_channels()),
                      dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            if self.channel_wise:
                im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255

        return im.astype(dtype=np.uint8)

    def render_shapes(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_n_channels()),
                      dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            shape_id = idx % 8
            if shape_id == 0:
                rr, cc = circle(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 1:
                rr, cc = triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 2:
                rr, cc = square(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 3:
                rr, cc = parallelogram(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 4:
                rr, cc = cross(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 5:
                rr, cc = diamond(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            elif shape_id == 6:
                rr, cc = pentagon(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            else:
                rr, cc = scalene_triangle(
                    pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)

            if self.channel_wise:
                im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255
        return im.astype(np.uint8)


class AdHocPushAgent:
    def __init__(self, env: Push, random_action_proba=0.5):
        self.env = None
        self.random_action_proba = random_action_proba
        self.set_env(env)

    def set_env(self, env: Push):
        self.env = env
        assert len(env.passive_box_ids) == 0
        assert not env.do_reward_push_only
        assert env.do_reward_active_box

    def act(self, observation, reward, done):
        if random.random() < self.random_action_proba:
            return self.env.action_space.sample()

        box_pos_in_game = [(idx, box_pos) for idx, box_pos in enumerate(self.env.box_pos)
                           if idx not in self.env.goal_ids and idx not in self.env.static_box_ids and box_pos[0] != -1]
        idx, box_pos = random.choice(box_pos_in_game)
        goal_pos = self.env.box_pos[next(iter(self.env.goal_ids))]
        delta = goal_pos - box_pos
        if np.abs(delta)[0] >= np.abs(delta)[1]:
            direction = (int(delta[0] > 0) * 2 - 1, 0)
        else:
            direction = (0, int(delta[1] > 0) * 2 - 1)

        return idx * 4 + self.env.direction2action[direction]


class RandomPushAgent:
    def __init__(self, env: Push):
        self.env = None
        self.set_env(env)

    def set_env(self, env: Push):
        self.env = env

    def act(self, observation, reward, done):
        return env.action_space.sample()


if __name__ == "__main__":
    """
    If called directly with argument "random", evaluates the average return of a random policy.
    If called without arguments, starts an interactive game played with wasd to move, q to quit.
    """
    env = Push(n_active_boxes=3, n_passive_boxes=1, n_goals=1, width=5, render_scale=10, channel_wise=False,
               return_state=False, observation_type='shapes', max_episode_steps=75, channels_first=False,
               ternary_interactions=True, do_reward_push_only=False, do_reward_active_box=True, )

    if len(sys.argv) > 1:
        if sys.argv[1] == "random":
            agent = RandomPushAgent(env)
        elif sys.argv[1] == "ad_hoc":
            proba = float(sys.argv[2])
            agent = AdHocPushAgent(env, random_action_proba=proba)
        else:
            raise ValueError(f'Unexpected agent type: {sys.argv[1]}')

        print(f'{sys.argv[1]} agent in action')

        all_r = []
        all_l = []
        n_episodes = 1000
        for i in range(n_episodes):
            if i % 100 == 0:
                print(i)

            s = env.reset()
            done = False
            episode_r = 0
            l = 0
            while not done:
                a = agent.act(None, None, None)
                s, r, done, _ = env.step(a)
                episode_r += r
                l += 1
            all_r.append(episode_r)
            all_l.append(l)
        print(f'Total reward: {np.mean(all_r)} +/- {np.std(all_r)}, std_mean={np.std(all_r) / np.sqrt(n_episodes)}')
        print(f'Length: {np.mean(all_l)} +/- {np.std(all_l)}, std_mean={np.std(all_l) / np.sqrt(n_episodes)}')
    else:
        s = env.reset()
        plt.imshow(s)
        plt.show()
        env.print()
        episode_r = 0

        while True:
            keys = list(input())
            if keys[0] == "q":
                break

            obj_id = int(keys[0])
            key = keys[1]
            print(f'obj_id:{obj_id},action={key}')
            if key == "a":
                a = 1
            elif key == "s":
                a = 0
            elif key == "d":
                a = 3
            elif key == "w":
                a = 2
            else:
                raise ValueError('Invalid action key:', key)

            a += 4 * obj_id

            s, r, d, _ = env.step(a)
            episode_r += r
            env.print(f'. Episode reward: {episode_r}')
            plt.imshow(s)
            plt.show()
            if d or key == "r":
                print("Done with {} points. Resetting!".format(episode_r))
                s = env.reset()
                episode_r = 0
                env.print()
                plt.imshow(s)
                plt.show()
