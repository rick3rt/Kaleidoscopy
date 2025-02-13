# %% importts

import numpy as np
import cv2
import matplotlib.pyplot as plt

from kaleidoscope.controller import Kaleidoscope, KSettings, KaleidoscopeController
from kaleidoscope.utils.plot import show_img_grid
from kaleidoscope.utils.export import write_video
from kaleidoscope.image import Image


# %% functions


# %%

k_conf1 = KSettings(
    num_repeats=3,
    angle_offset_in=0,
    angle_offset_out=0,
    # center_in=(100, 200),
    # pad_size=None,
)

k_conf2 = KSettings(
    num_repeats=3,
    angle_offset_in=np.pi / 3,
    angle_offset_out=0,
    # center_in=(100, 200),
    # pad_size=None,
)


filename = "image/image_doppler.png"


image_input = Image.load(filename)


kal1 = Kaleidoscope(
    k_conf1,
    image_input,
)

kal2 = Kaleidoscope(
    k_conf2,
    kal1.out(),
)

images = {
    "original": image_input,
    "blank": np.zeros_like(image_input),
    "k1": kal1.out(),
    "k2": kal2.out(),
}

show_img_grid(images)

# %%
from kaleidoscope.generator import create_pattern, kaleidoscope_pattern
from kaleidoscope.utils.plot import show_img, show_img_grid

# kal1 = Kaleidoscope(
#     k_conf1,
#     image_input,
# )


def plot_mapping(mapping: list | tuple, figsize=(10, 5), ncols=2, colorbar=False):
    n = len(mapping)
    nrows = n // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()
    for i, m in enumerate(mapping):
        axs[i].imshow(m)
        if colorbar:
            plt.colorbar(axs[i].imshow(m), ax=axs[i], shrink=0.6)
    plt.show()


class TMP:
    def __init__(self, settings, img):
        self.settings = settings
        self.image_in = img.copy()
        self.image_out = None

    def apply_pattern(self, pix_mag, pix_theta_k):
        self.image_out = kaleidoscope_pattern(
            self.image_in,
            pix_mag,
            pix_theta_k,
            self.settings["center_in"],
            self.settings["invert_sine"],
        )
        return self.image_out

    def create_pattern(self):

        pix_mag, pix_theta_k = create_pattern(
            self.image_in,
            self.settings["num_repeats"],
            self.settings["angle_offset_in"],
            self.settings["angle_offset_out"],
            self.settings["center_in"],
            self.settings["center_out"],
        )

        return pix_mag, pix_theta_k
        # # crop output image
        # if self.settings["pad_size"]:
        #     pix_mag = img_center_crop(pix_mag, *self.settings["pad_size"])
        #     pix_theta_k = img_center_crop(pix_theta_k, *self.settings["pad_size"])
        #     tmp = img_center_crop(self.image_in, *self.settings["pad_size"])
        # else:
        #     tmp = self.image_in

        # def stacker(input, size=(1, 3), axis=0):
        #     input = np.tile(input, size)
        #     input = np.concatenate([np.flip(input), input], axis=axis)
        #     input = np.concatenate([np.flip(input), input], axis=axis)
        #     return input

        # grid_width = 4
        # pix_mag = np.tile(pix_mag, (4, grid_width))
        # pix_theta_k = stacker(pix_theta_k, size=(1, grid_width), axis=0)
        # tmp = np.tile(tmp, (4, grid_width, 1))


k_conf1 = KSettings(
    num_repeats=2,
    angle_offset_in=0,
    angle_offset_out=0,
    # center_in=(100, 200),
    # pad_size=None,
)

k_conf2 = KSettings(
    num_repeats=3,
    angle_offset_in=1.5,
    angle_offset_out=0,
    # center_in=(100, 200),
    # pad_size=None,
)


filename = "image/image_test.png"
filename = "image/image_doppler.png"
image_input = Image.load(filename)

tmp = TMP(k_conf1, image_input)

pix_mag, pix_theta_k = tmp.create_pattern()
img_kal1 = tmp.apply_pattern(pix_mag, pix_theta_k)

# %%
# plot_mapping(
#     (pix_mag, np.rad2deg(pix_theta_k), image_input, img_kal1), ncols=2, colorbar=True
# )


def stacker(input, size=(1, 3), axis=0):
    input = np.tile(input, size)
    input = np.concatenate([np.flip(input), input], axis=axis)
    input = np.concatenate([np.flip(input), input], axis=axis)
    return input


def stacker2(input, size=(1, 1)):

    input = np.concatenate([np.flip(input, 1), input], axis=1)
    # input = np.concatenate([input, input], axis=1)
    input = np.concatenate([input, np.flip(input, 0)], axis=0)
    # input = np.concatenate([input, np.flip(input, 0)], axis=0)

    if input.ndim == 3:
        size = (size[0], size[1], 1)
    input = np.tile(input, size)

    return input


# pix_mag = stacker2(pix_mag)
# pix_theta_k = stacker2(pix_theta_k)


# pix_theta_k = np.pow(pix_theta_k, 1.5)
# pix_theta_k = pix_theta_k + map(pix_mag, 0, np.max(pix_mag[:]), -1, 0.5)
# pix_theta_k = pix_theta_k % angle_range
# pix_theta_k = pix_theta_k % (2 * np.pi)


def mag_mapper(mag):
    h, w = pix_mag.shape[:2]
    diag = np.sqrt(h**2 + w**2) // 2

    mag = mag / diag  # scale between 0 and 1

    mag = np.power(mag, 1.5)
    # scale between ... and 1
    # mag = mag / np.max(mag)

    mag = mag * diag  # scale back to original size
    return mag


def theta_mapper(angle):
    max_angle = np.max(angle)
    angle = angle / max_angle  # scale between 0 and 1

    angle = np.power(angle, 1.7)
    # scale between ... and 1
    angle = angle / np.max(angle)

    angle = angle * max_angle  # scale back to original size
    return angle


k_conf1 = KSettings(
    num_repeats=6,
    angle_offset_in=1,
    angle_offset_out=0,
    # pad_size=400,
    grid_size=None,
    mag_mapper=mag_mapper,
    angle_mapper=theta_mapper,
)


k_conf2 = KSettings(
    num_repeats=6,
    angle_offset_in=1.5,
    angle_offset_out=1.50,
    # center_in=(200, 200),
    # center_out=(300, 120),
    pad_size=None,
    grid_size=(2, 2),
    angle_mapper=theta_mapper,
)


mapper1 = KaleidoscopeMapper(
    k_conf1,
    image_input,
)
image_kal1 = mapper1.run()

mapper2 = KaleidoscopeMapper(
    k_conf2,
    image_kal1,
)
image_kal2 = mapper2.run()

images = {
    "original": image_input,
    "blank": np.zeros_like(image_input),
    "k1": image_kal1,
    "k2": image_kal2,
}

show_img_grid(images)

# %%
# pix_mag_mapped = mag_mapper(pix_mag)
# theta_k_mapped = theta_mapper(pix_theta_k)

# plot_mapping(
#     (pix_mag, pix_mag_mapped, pix_theta_k, theta_k_mapped), ncols=2, colorbar=True
# )


# x = np.linspace(0, diag)
# y = mag_mapper(x)
# plt.plot(x, y)
# mag_mapper = None
# img = img_kal1
img = image_input
mapper = KaleidoscopeMapper(k_conf2, img)
mapper.create_map()

pix_mag, pix_theta_k = mapper.mapping()


# img_rep = stacker2(image_input, size=grid_size)
tmp2 = TMP(k_conf2, image_input)

# or use tmp2
# pix_mag, pix_theta_k = tmp2.create_pattern()
# pix_mag = stacker2(pix_mag, size=grid_size)
# pix_theta_k = stacker2(pix_theta_k, size=grid_size)


plot_mapping((pix_mag, np.rad2deg(pix_theta_k)), ncols=2, colorbar=True)


# img_kal2 = tmp2.apply_pattern(pix_mag, pix_theta_k)

# images = {
#     "original": image_input,
#     "blank": np.zeros_like(image_input),
#     "k1": img_kal1,
#     "k2": img_kal2,
# }
# show_img_grid(images)


# %%
# show_img(image_kal2)

# Image.save("output/very_large4.png", image_kal2)
#


# %%


k_conf1 = KSettings(
    num_repeats=6,
    angle_offset_in=1,
    angle_offset_out=0,
    # pad_size=400,
    grid_size=None,
    mag_mapper=mag_mapper,
    angle_mapper=theta_mapper,
)

mapper1 = KaleidoscopeMapper(
    k_conf1,
    image_input,
)
image_kal1 = mapper1.run()

pix_max, pix_theta_k = mapper1.mapping()

plot_mapping((pix_max, np.rad2deg(pix_theta_k)), ncols=2, colorbar=True)

images = {
    "original": image_input,
    "blank": np.zeros_like(image_input),
    "k1": image_kal1,
    # "k2": image_kal2,
}

# show_img_grid(images)
