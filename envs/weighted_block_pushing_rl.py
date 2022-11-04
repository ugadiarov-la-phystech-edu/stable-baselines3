"""Gym environment for block pushing tasks (2D Shapes and 3D Cubes)."""
import numpy as np

import gym
from collections import OrderedDict
from dataclasses import dataclass
from gym import spaces
from gym.utils import seeding

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import skimage
import skimage.transform

from envs.utils import get_colors_and_weights

mpl.use('agg')

def diamond(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width // 2, r0 + width, r0 + width // 2], [c0 + width // 2, c0, c0 + width // 2, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def triangle(r0, c0, width, im_size):
    rr = np.asarray([r0 + 4] * 5 + [r0 + 3] * 3  + [r0 + 2] * 3 + [r0 + 1] + [r0], dtype=np.int32)
    cc = np.asarray(list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) * 2 + [c0 + 2] * 2, dtype=np.int32)
    # rr, cc = [r0, r0 + width - 1, r0 + width - 1], [c0 + width // 2, c0, c0 + width - 1]
    # return skimage.draw.polygon(rr, cc, im_size)
    return rr, cc

def circle(r0, c0, width, im_size):
    rr = np.asarray([r0] + [r0 + 1] * 3 + [r0 + 2] * 5 + [r0 + 3] * 3 + [r0 + 4], dtype=np.int32)
    cc = np.asarray([c0 + 2] + list(range(c0 + 1, c0 + 4)) + list(range(c0, c0 + 5)) + list(range(c0 + 1, c0 + 4)) + [c0 + 2], dtype=np.int32)
    # rr, cc = [r0, r0 + width - 1, r0 + width - 1], [c0 + width // 2, c0, c0 + width - 1]
    # return skimage.draw.polygon(rr, cc, im_size)
    return rr, cc

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
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0 + width // 2, c0 + width, c0 + width - width // 2]
    return skimage.draw.polygon(rr, cc, im_size)


def scalene_triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width//2], [c0 + width - width // 2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)

def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(objects, width):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ['purple', 'green', 'orange', 'blue', 'brown']

    for i, pos in objects.items():
        voxels[pos.pos.x, pos.pos.y, 0] = True
        colors[pos.pos.x, pos.pos.y, 0] = cols[i]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))
    return im


@dataclass
class Coord:
    x: int
    y: int

    def __add__(self, other):
        return Coord(self.x + other.x,
                     self.y + other.y)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__


@dataclass
class Object:
    pos: Coord
    weight: int


class InvalidMove(BaseException):
    pass


class InvalidPush(BaseException):
    pass


class BlockPushingRL(gym.Env):
    """Gym environment for block pushing task."""

    def __init__(self, width=10, height=10, render_type='cubes',
                 *, num_objects=3, scale=5, mode='Train', cmap='Set1', typ='Observed',
                 num_weights=None, seed=None, observation_full_state=False):
        self.observation_full_state = observation_full_state
        self.width = width
        self.height = height
        self.render_type = render_type
        self.mode = mode
        self.cmap = cmap
        self.typ = typ
        self.scale = scale
        self.new_colors = None

        if typ in ['Unobserved', 'FixedUnobserved'] and "FewShot" in mode:
            self.n_f = int(mode[-1])
            if cmap == 'Sets':
                self.new_colors = np.random.choice(12, self.n_f, replace=False)
            elif cmap == 'Pastels':
                self.new_colors = np.random.choice(8, self.n_f, replace=False)
            else:
                print("something went wrong")

        self.num_objects = num_objects
        self.num_actions = 5 * self.num_objects  # Move StayNESW
        if num_weights is None:
            num_weights = num_objects
        self.num_weights = num_weights

        self.np_random = None
        self.game = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = OrderedDict()
        self.target_objects = OrderedDict()

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.width * self.scale, self.height * self.scale, 6),
            dtype=np.uint8
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_grid(self, objects):
        im = np.zeros((3, self.width, self.height))
        for idx, obj in objects.items():
            im[:, obj.pos.x, obj.pos.y] = self.colors[idx][:3]
        return im

    def render_circles(self, objects):
        im = np.zeros((self.width * self.scale, self.height * self.scale, 3), dtype=np.float32)
        radius = self.scale // 2
        for idx, obj in objects.items():
            rr, cc = skimage.draw.circle(
                obj.pos.x * self.scale + radius, obj.pos.y * self.scale + radius, radius, im.shape)
            im[rr, cc, :] = self.colors[idx][:3]

        im *= 255
        return im.astype(np.uint8)

    def render_shapes(self, objects):
        im = np.zeros((self.width * self.scale, self.height * self.scale, 3), dtype=np.float32)
        for idx, obj in objects.items():
            if self.shapes[idx] == 0:
                # radius = self.scale // 2
                # rr, cc = skimage.draw.circle(
                #     obj.pos.x * self.scale + radius, obj.pos.y * self.scale + radius, radius, im.shape)
                rr, cc = circle(obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            elif self.shapes[idx] == 1:
                rr, cc = triangle(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            elif self.shapes[idx] == 2:
                rr, cc = square(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            elif self.shapes[idx] == 3:
                rr, cc = diamond(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            elif self.shapes[idx] == 4:
                rr, cc = cross(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            elif self.shapes[idx] == 5:
                rr, cc = pentagon(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            elif self.shapes[idx] == 6:
                rr, cc = parallelogram(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)
            else:
                rr, cc = scalene_triangle(
                    obj.pos.x * self.scale, obj.pos.y * self.scale, self.scale, im.shape)

            im[rr, cc, :] = self.colors[idx][:3]

        im *= 255
        return im.astype(np.uint8)

    def render_cubes(self, objects):
        im = render_cubes(objects, self.width)
        return im.astype(np.uint8)

    def _get_observation(self):
        if self.observation_full_state:
            return self.get_state(), self.render()
        else:
            return self.render()

    def render(self):
        render = dict(
            grid=self.render_grid,
            circles=self.render_circles,
            shapes=self.render_shapes,
            cubes=self.render_cubes,
        )[self.render_type]

        objects_image = render(self.objects)
        target_image = render(self.target_objects)
        return np.concatenate([objects_image, target_image], axis=-1)

    def get_state(self):
        im = np.zeros(
            (self.width, self.height, self.num_objects * 2), dtype=np.int32)
        for idx, obj in self.objects.items():
            im[obj.pos.x, obj.pos.y, idx] = 1
            target_obj = self.target_objects[idx]
            im[target_obj.pos.x, target_obj.pos.y, idx + self.num_objects] = 1
        return im

    def get_sparse_reward(self):
        num_hits = len([idx for idx, obj in self.objects.items() if obj.pos == self.target_objects[idx].pos])

        return num_hits / self.num_objects

    def get_dense_reward(self):
        distance = 0.0
        for i in range(self.num_objects):
            distance += np.abs(self.objects[i].pos.x - self.target_objects[i].pos.x) +\
                        np.abs(self.objects[i].pos.y - self.target_objects[i].pos.y)

        return -distance / self.num_objects

    def _sample_positions(self, n):
        locations = np.random.choice(self.width * self.height, n, replace=False)
        xs, ys = np.unravel_index(locations, [self.width, self.height])
        return list(zip(xs, ys))

    def reset(self):
        if self.typ == 'FixedUnobserved' or self.typ == 'Observed':
            self.shapes = np.arange(self.num_objects)
        elif self.mode == 'ZeroShotShape':
            self.shapes = np.random.choice(6, self.num_objects)
        else:
            self.shapes = np.random.choice(3, self.num_objects)

        self.objects = OrderedDict()
        self.target_objects = OrderedDict()
        if self.typ == 'Observed':
            self.colors, weights = get_colors_and_weights(
                cmap=self.cmap,
                num_colors=self.num_objects,
                observed=True,
                mode=self.mode,
                randomize=False
            )
        else:
            self.colors, weights = get_colors_and_weights(
                cmap=self.cmap,
                num_colors=self.num_objects,
                observed=False,
                mode=self.mode,
                new_colors=self.new_colors)

        # Randomize object position.
        for idx, position in enumerate(self._sample_positions(self.num_objects)):
            self.objects[idx] = Object(pos=Coord(x=position[0], y=position[1]), weight=weights[idx])
        for idx, position in enumerate(self._sample_positions(self.num_objects)):
            self.target_objects[idx] = Object(pos=Coord(x=position[0], y=position[1]), weight=None)

        return self._get_observation()

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos.x not in range(0, self.width):
            return False
        if pos.y not in range(0, self.height):
            return False

        if self.collisions:
            for idx, obj in self.objects.items():
                if idx == obj_id:
                    continue

                if pos == obj.pos:
                    return False

        return True

    def valid_move(self, obj_id, offset: Coord):
        """Check if move is valid."""
        old_obj = self.objects[obj_id]
        new_pos = old_obj.pos + offset
        return self.valid_pos(new_pos, obj_id)

    def occupied(self, pos: Coord):
        for idx, obj in self.objects.items():
            if obj.pos == pos:
                return idx
        return None

    def translate(self, obj_id, offset: Coord, n_parents=0):
        """"Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) coordinate.
        """
        obj = self.objects[obj_id]

        other_object_id = self.occupied(obj.pos + offset)
        if other_object_id is not None:
            if n_parents == 1:
                # cannot push two objects
                raise InvalidPush()
            if obj.weight > self.objects[other_object_id].weight:
                self.translate(other_object_id, offset,
                               n_parents=n_parents+1)
            else:
                raise InvalidMove()
        if not self.valid_move(obj_id, offset):
            raise InvalidMove()

        self.objects[obj_id] = Object(
            pos=obj.pos+offset, weight=obj.weight)

    def step(self, action: int):
        directions = [Coord(0, 0),
                      Coord(-1, 0),
                      Coord(0, 1),
                      Coord(1, 0),
                      Coord(0, -1)]

        direction = action % 5
        obj_id = action // 5

        info = {'invalid_push': False}
        try:
            self.translate(obj_id, directions[direction])
        except InvalidMove:
            pass
        except InvalidPush:
            info['invalid_push'] = True

        img = self._get_observation()

        # reward = self.get_sparse_reward(self.target[0])
        reward = self.get_dense_reward()
        num_hits = len([idx for idx, obj in self.objects.items() if obj.pos == self.target_objects[idx].pos])
        done = (num_hits == self.num_objects)

        return img, reward, done, info

    def sample_step(self, action: int):
        directions = [Coord(0, 0),
                      Coord(-1, 0),
                      Coord(0, 1),
                      Coord(1, 0),
                      Coord(0, -1)]

        direction = action % 5
        obj_id = action // 5
        done = False
        info = {'invalid_push': False}

        objects = self.objects.copy()
        try:
            self.translate(obj_id, directions[direction])
        except InvalidMove:
            pass
        except InvalidPush:
            info['invalid_push'] = True

        reward = self.get_dense_reward()
        next_obs = self._get_observation()
        self.objects = objects

        return reward, next_obs

    def get_target(self, num_steps):
        objects = self.objects.copy()

        for i in range(num_steps):
            move = np.random.choice(self.num_objects * 5)
            state, _, _, _ = self.step(move)

        self.target_objects = self.objects.copy()
        self.target = state
        self.objects = objects
