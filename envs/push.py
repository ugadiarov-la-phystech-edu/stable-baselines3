import gym
import numpy as np
import skimage
from gym.utils import seeding
from gym import spaces
# from scipy.misc import imresize
import sys
import copy

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
    STATIC_BOX = 'static_box'
    BOX = 'box'

    def __init__(self, mode='default', n_boxes=5, n_static_boxes=0, n_goals=1, static_goals=True, width=10,
                 embodied_agent=False, return_state=True, observation_type='squares', max_episode_steps=75,
                 hard_walls=False, channels_first=True, seed=None, render_scale=5):
        self.w = width
        self.step_limit = max_episode_steps
        self.n_boxes = n_boxes
        self.embodied_agent = embodied_agent

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

        self.n_boxes_in_game = self.n_boxes - len(self.goal_ids) - len(self.static_box_ids) - self.embodied_agent
        self.static_goals = static_goals
        self.render_scale = render_scale
        self.hard_walls = hard_walls
        self.colors = get_colors(num_colors=max(9, self.n_boxes))
        self.observation_type = observation_type
        self.return_state = return_state
        self.channels_first = channels_first

        self.directions = {
            0: np.asarray((1, 0)),
            1: np.asarray((0, -1)),
            2: np.asarray((-1, 0)),
            3: np.asarray((0, 1))
        }

        self.np_random = None

        if self.embodied_agent:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Discrete(
                4 * (self.n_boxes - len(self.goal_ids) * self.static_goals - len(self.static_box_ids))
            )

        if self.observation_type == 'grid':
            raise NotImplementedError(f'Observation type "{observation_type}" has not been implemented yet')
            # channels are movable boxes, goals, static boxes
            self.observation_space = spaces.Box(
                0,
                1,
                (self.w, self.w, self.n_boxes)
            )
        elif self.observation_type in ('squares', 'shapes'):
            observation_shape = (self.w * self.render_scale, self.w * self.render_scale, 3)
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
            return self.state, image

        return image

    def reset(self):
        state = np.full(shape=[self.w, self.w], fill_value=-1, dtype=np.int32)

        # sample random locations for self, goals, and walls.
        locs = np.random.choice(self.w ** 2, self.n_boxes, replace=False)

        xs, ys = np.unravel_index(locs, [self.w, self.w])

        # populate state with locations
        for i, (x, y) in enumerate(zip(xs, ys)):
            state[x, y] = i
            self.box_pos[i, :] = x, y

        self.state = state
        self.steps_taken = 0

        return self._get_observation()

    def _get_type(self, box_id):
        if box_id in self.goal_ids:
            return self.GOAL
        elif box_id in self.static_box_ids:
            return self.STATIC_BOX
        else:
            return self.BOX

    def _destroy_box(self, box_id):
        box_pos = self.box_pos[box_id]
        self.state[box_pos[0], box_pos[1]] = -1
        self.box_pos[box_id] = -1, -1
        if self._get_type(box_id) == self.BOX:
            self.n_boxes_in_game -= 1

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

        vec = self.directions[action % len(self.directions)]
        box_id = action // len(self.directions)
        box_old_pos = self.box_pos[box_id]
        box_new_pos = box_old_pos + vec
        box_type = self._get_type(box_id)

        done = False
        reward = -0.01

        if self._is_free_cell(box_old_pos):
            # This box is out of the game. There is nothing to do.
            pass
        elif not self.is_in_grid(box_new_pos):
            reward += -0.1
            if not self.hard_walls:
                # push out of grid, destroy box or finish episode if an agent is out of the grid
                if self.embodied_agent:
                    reward -= 1
                    done = True
                else:
                    self._destroy_box(box_id)
        elif not self._is_free_cell(box_new_pos):
            # push into another box
            another_box_id = self._get_occupied_box_id(box_new_pos)
            another_box_type = self._get_type(another_box_id)

            if box_type == self.BOX:
                if another_box_type == self.BOX:
                    another_box_new_pos = box_new_pos + vec
                    if self.is_in_grid(another_box_new_pos):
                        if self._is_free_cell(another_box_new_pos):
                            self._move(another_box_id, another_box_new_pos)
                            self._move(box_id, box_new_pos)
                        elif self._get_type(self._get_occupied_box_id(another_box_new_pos)) == self.GOAL:
                            reward += 1
                            self._destroy_box(another_box_id)
                            self._move(box_id, box_new_pos)
                        else:
                            reward += -0.1
                    else:
                        reward += -0.1
                        if not self.hard_walls:
                            self._destroy_box(another_box_id)
                            self._move(box_id, box_new_pos)
                elif another_box_type == self.GOAL:
                    if self.embodied_agent:
                        another_box_new_pos = box_new_pos + vec
                        if self.is_in_grid(another_box_new_pos):
                            if self._is_free_cell(another_box_new_pos):
                                self._move(another_box_id, another_box_new_pos)
                                self._move(box_id, box_new_pos)
                            elif self._get_type(self._get_occupied_box_id(another_box_new_pos)) == self.GOAL:
                                reward += -0.1
                            else:
                                assert self._get_type(self._get_occupied_box_id(another_box_new_pos)) in (self.BOX, self.STATIC_BOX)
                                reward += 1
                                self._destroy_box(self._get_occupied_box_id(another_box_new_pos))
                                self._move(another_box_id, another_box_new_pos)
                                self._move(box_id, box_new_pos)
                        else:
                            reward += -0.1
                            if not self.hard_walls:
                                reward -= 1
                                self._destroy_box(another_box_id)
                                self._move(box_id, box_new_pos)
                    else:
                        reward += 1
                        self._destroy_box(box_id)
                else:
                    assert box_type == self.STATIC_BOX
                    reward += -0.1
            elif box_type == self.GOAL:
                if another_box_type in (self.BOX, self.STATIC_BOX):
                    self._destroy_box(another_box_id)
                    self._move(box_id, box_new_pos)
                    reward += 1
                else:
                    assert another_box_type == self.GOAL
                    reward += -0.1
            else:
                assert False, f'Cannot move a box of type:{box_type}'
        else:
            # pushed into open space, move box
            self._move(box_id, box_new_pos)

        self.steps_taken += 1
        if self.steps_taken >= self.step_limit:
            done = True

        if self.n_boxes_in_game == 0:
            done = True

        return self._get_observation(), reward, done, {}

    def is_in_grid(self, point):
        return (0 <= point[0] < self.w) and (0 <= point[1] < self.w)

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
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, 3), dtype=np.float32)
        for idx, pos in enumerate(self.box_pos):
            if pos[0] == -1:
                assert pos[1] == -1
                continue

            rr, cc = square(pos[0] * self.render_scale, pos[1] * self.render_scale, self.render_scale, im.shape)
            im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255

        return im.astype(dtype=np.uint8)

    def render_shapes(self):
        im = np.zeros((self.w * self.render_scale, self.w * self.render_scale, 3), dtype=np.float32)
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

            im[rr, cc, :] = self.colors[idx][:3]

        if self.channels_first:
            im = im.transpose([2, 0, 1])

        im *= 255
        return im.astype(np.uint8)


if __name__ == "__main__":
    """
    If called directly with argument "random", evaluates the average return of a random policy.
    If called without arguments, starts an interactive game played with wasd to move, q to quit.
    """
    env = Push(n_boxes=5, n_static_boxes=0, n_goals=1, static_goals=True, observation_type='shapes', hard_walls=True,
               channels_first=False, width=5, embodied_agent=True, render_scale=10)

    if len(sys.argv) > 1 and sys.argv[1] == "random":
        all_r = []
        n_episodes = 1000
        for i in range(n_episodes):
            s = env.reset()
            plt.imshow(s[1])
            plt.show()
            done = False
            episode_r = 0
            while not done:
                s, r, done, _ = env.step(np.random.randint(env.action_space.n))
                episode_r += r
                plt.imshow(s[1])
                plt.show()
            all_r.append(episode_r)
        print(np.mean(all_r), np.std(all_r), np.std(all_r) / np.sqrt(n_episodes))
    else:
        s = env.reset()
        plt.imshow(s[1])
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
            env.print(f'. Reward: {episode_r}')
            plt.imshow(s[1])
            plt.show()
            if d or key == "r":
                print("Done with {} points. Resetting!".format(episode_r))
                s = env.reset()
                episode_r = 0
                env.print()