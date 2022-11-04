import numpy as np
from matplotlib import pyplot as plt


def observed_colors(num_colors, mode, randomize=True):
    if mode == 'ZeroShot':
        c = np.sort(np.random.uniform(0.0, 1.0, size=num_colors))
    else:
        c = (np.arange(num_colors)) / (num_colors-1)
        if not randomize:
            return c

        diff = 1.0 / (num_colors - 1)
        if mode == 'Train':
            diff = diff / 8.0
        elif mode == 'Test-v1':
            diff = diff / 4.0
        elif mode == 'Test-v2':
            diff = diff / 3.0
        elif mode == 'Test-v3':
            diff = diff / 2.0

        unif = np.random.uniform(-diff+EPS, diff-EPS, size=num_colors)
        unif[0] = abs(unif[0])
        unif[-1] = -abs(unif[-1])

        c = c + unif

    return c


def get_cmap(cmap, mode):
    length = 9
    if cmap == 'Sets':
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Set1')
        else:
            cmap = [plt.get_cmap('Set1'), plt.get_cmap('Set3')]
            length = [9,12]
    else :
        if "FewShot" not in mode:
            cmap = plt.get_cmap('Pastel1')
        else:
            cmap = [plt.get_cmap('Pastel1'), plt.get_cmap('Pastel2')]
            length = [9,8]

    return cmap, length


def unobserved_colors(cmap, num_colors, mode, new_colors=None):
    if mode in ['Train', 'ZeroShotShape']:
        cm, length = get_cmap(cmap, mode)
        weights = np.sort(np.random.choice(length, num_colors, replace=False))
        colors = [cm(i/length) for i in weights]
    else:
        cm, length = get_cmap(cmap, mode)
        cm1, cm2 = cm
        length1, length2 = length
        l = length1 + len(new_colors)
        w = np.sort(np.random.choice(l, num_colors, replace=False))
        colors = []
        weights = []
        for i in w:
            if i < length1:
                colors.append(cm1(i/length1))
                weights.append(i)
            else:
                colors.append(cm2(new_colors[i - length1] / length2))
                weights.append(new_colors[i - length1] + 0.5)

    return colors, weights


def get_colors_and_weights(cmap='Set1', num_colors=9, observed=True,
    mode='Train', new_colors=None, randomize=True):
    """Get color array from matplotlib colormap."""
    if observed:
        c = observed_colors(num_colors, mode, randomize=randomize)
        cm = plt.get_cmap(cmap)

        colors = []
        for i in reversed(range(num_colors)):
            colors.append((cm(c[i])))

        weights = [num_colors - idx
                       for idx in range(num_colors)]
    else:
        colors, weights = unobserved_colors(cmap, num_colors, mode, new_colors)

    return colors, weights