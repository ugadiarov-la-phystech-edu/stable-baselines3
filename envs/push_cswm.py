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
    AGENT = 'agent'
    GOAL = 'goal'
    STATIC_BOX = 'static_box'
    BOX = 'box'
    MOVING_BOXES_KEY = 'moving_boxes'

    STEP_REWARD = -0.01
    OUT_OF_FIELD_REWARD = -0.1
    COLLISION_REWARD = -0.1
    DEATH_REWARD = -1
    HIT_GOAL_REWARD = 1
    DESTROY_GOAL_REWARD = -1

    def __init__(self, n_boxes=5, n_static_boxes=0, n_goals=1, static_goals=True, width=5,
                 embodied_agent=False, return_state=True, observation_type='shapes', max_episode_steps=75,
                 border_walls=True, channels_first=True, channel_wise=False, channels_for_static_objects=True,
                 seed=None, render_scale=10,
                 ):
        if n_static_boxes > 0:
            assert n_goals == 0 or static_goals, 'Cannot have movable goals with static objects.'

        if n_goals > 0 and not static_goals:
            assert n_static_boxes == 0, 'Cannot have static objects with movable goals'

        self.w = width
        self.step_limit = max_episode_steps
        self.n_boxes = n_boxes
        self.embodied_agent = embodied_agent
        self.n_interactions_per_step = None

        self.goal_ids = set()
        self.static_box_ids = set()
        for k in range(self.n_boxes):
            box_id = self.n_boxes - k - 1
            if k < n_goals:
                self.goal_ids.add(box_id)
            elif k < n_goals + n_static_boxes:
                self.static_box_ids.add(box_id)
            else:
                break

        assert len(self.goal_ids) == n_goals
        assert len(self.static_box_ids) == n_static_boxes
        if self.embodied_agent:
            assert self.n_boxes > len(self.goal_ids) + len(self.static_box_ids)

        self.n_boxes_in_game = None
        self.static_goals = static_goals
        self.render_scale = render_scale
        self.border_walls = border_walls
        self.colors = get_colors(num_colors=max(9, self.n_boxes))
        self.observation_type = observation_type
        self.return_state = return_state
        self.channels_first = channels_first
        self.channel_wise = channel_wise
        self.channels_for_static_objects = channels_for_static_objects

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }
        self.direction2action = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}

        self.np_random = None

        if self.embodied_agent:
            self.action_space = spaces.Discrete(4)
        else:
            n_movable_objects = self.n_boxes - len(self.goal_ids) * self.static_goals - len(self.static_box_ids)
            self.action_space = spaces.Discrete(4 * n_movable_objects)

        if self.observation_type == 'grid':
            raise NotImplementedError(f'Observation type "{observation_type}" has not been implemented yet')
            # channels are movable boxes, goals, static boxes
            self.observation_space = spaces.Box(
                0,
                1,
                (self.w, self.w, self.n_boxes)
            )
        elif self.observation_type in ('squares', 'shapes'):
            observation_shape = (self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels())
            if self.channels_first:
                observation_shape = (observation_shape[2], *observation_shape[:2])
            self.observation_space = spaces.Box(0, 255, observation_shape, dtype=np.uint8)
        else:
            raise ValueError(f'Invalid observation_type: {self.observation_type}.')

        self.state = None
        self.steps_taken = 0
        self.pos = None
        self.image = None
        self.box_pos = np.zeros(shape=(self.n_boxes, 2), dtype=np.int32)

        self.seed(seed)
        self.reset()

    def _get_image_channels(self):
        if not self.channel_wise:
            return 3

        n_channels = self.n_boxes
        if not self.channels_for_static_objects:
            n_channels -= len(self.static_box_ids) + len(self.goal_ids) * self.static_goals

        return n_channels

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

    def reset(self):
        self._clear_state()

        # sample random locations for objects
        if self.embodied_agent:
            is_agent_in_main_area = self.np_random.random() > 4 * (self.w - 1) / self.w / self.w
            locs = self.np_random.choice((self.w - 2) ** 2, self.n_boxes - 1 + is_agent_in_main_area, replace=False)
            xs, ys = np.unravel_index(locs, [self.w - 2, self.w - 2])
            xs += 1
            ys += 1
            if not is_agent_in_main_area:
                agent_loc = self.np_random.choice(4 * (self.w - 1))
                side_id = agent_loc // (self.w - 1)
                if side_id == 0:
                    x = 0
                    y = agent_loc % (self.w - 1)
                elif side_id == 1:
                    x = agent_loc % (self.w - 1)
                    y = self.w - 1
                elif side_id == 2:
                    x = self.w - 1
                    y = self.w - 1 - agent_loc % (self.w - 1)
                elif side_id == 3:
                    x = self.w - 1 - agent_loc % (self.w - 1)
                    y = 0
                else:
                    raise ValueError(f'Unexpected side_id={side_id}')

                xs = np.append(x, xs)
                ys = np.append(y, ys)
        else:
            locs = self.np_random.choice(self.w ** 2, self.n_boxes, replace=False)
            xs, ys = np.unravel_index(locs, [self.w, self.w])

        # populate state with locations
        for i, (x, y) in enumerate(zip(xs, ys)):
            self._set_position(box_id=i, x=x, y=y)

        self.steps_taken = 0
        self.n_boxes_in_game = self.n_boxes - len(self.goal_ids) - int(self.embodied_agent)
        if not self.embodied_agent:
            self.n_boxes_in_game -= len(self.static_box_ids) * int(self.static_goals)

        return self._get_observation()

    def _clear_state(self):
        self.state = np.full(shape=[self.w, self.w], fill_value=-1, dtype=np.int32)
        self.box_pos = np.full(shape=(self.n_boxes, 2), fill_value=-1, dtype=np.int32)

    def _set_position(self, box_id, x, y):
        self.state[x, y] = box_id
        self.box_pos[box_id, :] = x, y

    def _is_agent(self, box_id):
        return self.embodied_agent and box_id == 0

    def _get_type(self, box_id):
        if box_id in self.goal_ids:
            return Push.GOAL
        elif box_id in self.static_box_ids:
            return Push.STATIC_BOX
        elif self._is_agent(box_id):
            return Push.AGENT
        else:
            return Push.BOX

    def _destroy_box(self, box_id):
        box_pos = self.box_pos[box_id]
        self.state[box_pos[0], box_pos[1]] = -1
        self.box_pos[box_id] = -1, -1
        if self._get_type(box_id) == Push.BOX or (not self.embodied_agent and self._get_type(box_id) == Push.STATIC_BOX):
            self.n_boxes_in_game -= 1

    def _move_one_box(self, box_id, new_pos):
        old_pos = self.box_pos[box_id]
        self.state[old_pos[0], old_pos[1]] = -1
        self.state[new_pos[0], new_pos[1]] = box_id
        self.box_pos[box_id] = new_pos

    def _move(self, box_id, displacement, n_interactions):
        box_type = self._get_type(box_id)
        box_old_pos = self.box_pos[box_id]
        box_new_pos = box_old_pos + displacement
        if not self._is_in_grid(box_new_pos, box_id):
            reward = Push.OUT_OF_FIELD_REWARD
            done = False
            movement = False
            if not self.border_walls:
                # push out of grid, destroy object or finish episode if an agent is out of the grid
                if box_type == Push.GOAL:
                    reward += Push.DESTROY_GOAL_REWARD
                elif box_type == Push.AGENT:
                    reward += Push.DEATH_REWARD
                    done = True
                movement = True
            return reward, done, [box_id], movement
        elif self._is_free_cell(box_new_pos):
            self._move_one_box(box_id, box_new_pos)
            reward = 0
            done = False
            movement = True
            return reward, done, [box_id], movement
        else:
            if self.n_interactions_per_step is not None and n_interactions >= self.n_interactions_per_step:
                # Reached the limit of interactions during one step
                reward = Push.COLLISION_REWARD
                done = False
                movement = False
                return reward, done, [box_id], movement

            # push into another box
            another_box_id = self._get_occupied_box_id(box_new_pos)
            another_box_type = self._get_type(another_box_id)
            if box_type == Push.AGENT:
                if another_box_type == Push.BOX:
                    reward, done, affected_boxes, movement = self._move(another_box_id, displacement, n_interactions + 1)
                    affected_boxes.append(box_id)
                    if movement:
                        self._move_one_box(box_id, box_new_pos)

                    return reward, done, affected_boxes, movement
                elif another_box_type in (Push.GOAL, Push.STATIC_BOX):
                    reward = Push.COLLISION_REWARD
                    done = False
                    movement = False
                    return reward, done, [box_id], movement
                else:
                    raise ValueError(f'Unexpected box type: {another_box_type}')
            elif box_type == Push.BOX:
                if another_box_type == Push.BOX:
                    reward, done, affected_boxes, movement = self._move(another_box_id, displacement, n_interactions + 1)
                    affected_boxes.append(box_id)
                    if movement:
                        self._move_one_box(box_id, box_new_pos)

                    return reward, done, affected_boxes, movement
                elif another_box_type == Push.GOAL:
                    self._destroy_box(box_id)
                    reward = Push.HIT_GOAL_REWARD
                    done = False
                    movement = True
                    return reward, done, [box_id], movement
                elif another_box_type == Push.STATIC_BOX:
                    reward = Push.COLLISION_REWARD
                    done = False
                    movement = False
                    return reward, done, [box_id], movement
                else:
                    raise ValueError(f'Unexpected box type: {another_box_type}')
            elif box_type == Push.GOAL:
                if another_box_type in (Push.BOX, Push.STATIC_BOX):
                    self._destroy_box(another_box_id)
                    self._move_one_box(box_id, box_new_pos)
                    reward = Push.HIT_GOAL_REWARD
                    done = False
                    movement =True
                    return reward, done, [box_id], movement
                elif another_box_type == Push.GOAL:
                    reward = Push.COLLISION_REWARD
                    done = False
                    movement = False
                    return reward, done, [box_id], movement
                else:
                    raise ValueError(f'Unexpected box type: {another_box_type}')
            else:
                raise ValueError(f'Unexpected box type: {box_type}')

    def _is_free_cell(self, pos):
        return self.state[pos[0], pos[1]] == -1

    def _get_occupied_box_id(self, pos):
        return self.state[pos[0], pos[1]]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        vec = self.directions[action % len(self.directions)]
        box_id = action // len(self.directions)
        reward, done, affected_boxes, movement = self._move(box_id, vec, n_interactions=0)
        reward += Push.STEP_REWARD

        moving_boxes = [0] * self.n_boxes
        for box_id in affected_boxes:
            moving_boxes[box_id] = 1

        self.steps_taken += 1
        info = {Push.MOVING_BOXES_KEY: moving_boxes}
        if self.steps_taken >= self.step_limit:
            info["TimeLimit.truncated"] = True
            done = True

        if self.n_boxes_in_game == 0:
            done = True

        return self._get_observation(), reward, done, info

    def _is_in_grid(self, point, box_id):
        if not self.embodied_agent or box_id == 0:
            return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

        return (1 <= point[0] < self.w - 1) and (1 <= point[1] < self.w - 1)

    def print(self, message=''):
        state = self.state
        chars = {-1: '.'}
        for box_id in range(self.n_boxes):
            if box_id in self.goal_ids:
                chars[box_id] = 'x'
            elif box_id in self.static_box_ids:
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
        return ["down", "left", "up", "right"] * (
                    self.n_boxes - len(self.static_box_ids) - len(self.goal_ids) * self.static_goals)

    def render_squares(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels()), dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            if self.channel_wise:
                if not self.channels_for_static_objects and (idx in self.goal_ids or idx in self.static_box_ids):
                    im[rr, cc] = 1
                else:
                    im[rr, cc, idx] = 1
            else:
                im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255

        return im.astype(dtype=np.uint8)

    def render_shapes(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, self._get_image_channels()), dtype=np.float32)
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
                if not self.channels_for_static_objects and (idx in self.goal_ids or idx in self.static_box_ids):
                    im[rr, cc] = 1
                else:
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
        assert not env.embodied_agent
        assert env.static_goals
        assert len(env.goal_ids) == 1
        assert len(env.static_box_ids) == 0

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
    env = Push(n_boxes=5, n_static_boxes=0, n_goals=1, static_goals=True, observation_type='shapes', border_walls=True,
               channels_first=False, width=7, embodied_agent=True, render_scale=10, return_state=False)

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

            s, r, d, info = env.step(a)
            episode_r += r
            env.print(f'. Episode reward: {episode_r}. Moving objects: {info[Push.MOVING_BOXES_KEY]}')
            plt.imshow(s)
            plt.show()
            if d or key == "r":
                print("Done with {} points. Resetting!".format(episode_r))
                s = env.reset()
                episode_r = 0
                env.print()
                plt.imshow(s)
                plt.show()
