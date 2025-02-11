import math
import numpy as np
from matplotlib import pyplot as plt


def show_img_grid(images_dict, ncols=2, show_axis=False, fig_size=(10, 10)):
    nrows = math.ceil(len(images_dict) / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=fig_size)
    if np.ndim(ax) == 1:
        ax = np.reshape(ax, [nrows, ncols])

    for i, (title, img) in enumerate(images_dict.items()):
        row, col = i // ncols, i % ncols
        ax[row, col].imshow(img)
        ax[row, col].set_title(title)
        if not show_axis:
            ax[row, col].axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def plot_point(p, *args, **kwargs):
    plt.plot(p[0], p[1], *args, **kwargs)


def plot_vec(start, end, *args, **kwargs):
    plt.plot([start[0], end[0]], [start[1], end[1]], *args, **kwargs)
