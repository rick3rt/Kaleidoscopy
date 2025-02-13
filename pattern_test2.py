# %% importts
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from kaleidoscope.controller import (
    Kaleidoscope,
    KSettings,
    KaleidoscopeController,
    KaleidoscopeMapper,
)
from kaleidoscope.utils.plot import show_img_grid, plot_mapping
from kaleidoscope.utils.export import write_video
from kaleidoscope.image import Image, copy_masked_region
import kaleidoscope.utils.pattern as pattern

# %% functions


# %%


def mag_mapper(mag):
    h, w = mag.shape[:2]
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


# k_conf2 = KSettings(
#     num_repeats=3,
#     angle_offset_in=np.pi / 3,
#     angle_offset_out=0,
#     # center_in=(100, 200),
#     # pad_size=None,
# )


filename = "image/image_doppler.png"
image_input = Image.load(filename)


kalm1 = KaleidoscopeMapper(k_conf1, image_input)
kalm1.run()
# kalm2 = KaleidoscopeMapper(k_conf2, kalm1.run())

m1_pix_mag, m1_pix_theta_k = kalm1.mapping()
# m2_pix_mag, m2_pix_theta_k = kalm2.mapping()


plot_mapping((m1_pix_mag, np.rad2deg(m1_pix_theta_k)), ncols=2, colorbar=True)

# %% pattern mappings

output_size = (2500, 2500)

# m1_pix_mag_pattern = pattern.pattern_hex(m1_pix_mag, 150, 0, output_size)
# m1_pix_theta_k_pattern = pattern.pattern_hex(m1_pix_theta_k, 150, 0, output_size)


print(kalm1.image_out.shape)

img_tmp = cv2.resize(kalm1.image_out, (500, 500))

img_pattern = pattern.pattern_hex(img_tmp, 300, 0, output_size)
img_pattern = img_pattern.astype(np.uint8)
print(img_pattern.shape)
print(img_pattern.dtype)
print(kalm1.image_out.shape)
print(kalm1.image_out.dtype)

# plot_mapping(
#     (m1_pix_mag, m1_pix_mag_pattern, m1_pix_theta_k, m1_pix_theta_k_pattern),
#     ncols=2,
#     colorbar=True,
# )

show_img_grid(
    {
        "original": image_input,
        "kal1": kalm1.image_out,
        "pattern": img_pattern,
        # "pattern_mag": m1_pix_mag_pattern,
    }
)

# %%
# create hexagonal mask

mask = np.zeros(image_input.shape[:2], dtype=np.uint8)
print(mask.shape)
# mask[0:10, :] = 1
# mask[:, 50:90] = 1


# create a function that will pack hexagons in a grid of specified size. also let the user specify the hexagon radius.
# return the vertices of the centers of the hegaxon
# also create a function that will create a hexagon mask given the vertices of the center of one hexagon


# %%
from kaleidoscope.utils.pattern import pack_hexagons, place_hexagon, place_hexagon_mask

hradius = 50
hangle = np.deg2rad(0)
hcenters = pack_hexagons((2, 2), hradius, hangle)

# print(hcenters)
# mask = np.zeros(image_input.shape[:2], dtype=np.uint8)
# for hc in hcenters:
#     hv = place_hexagon(hc, hradius, hangle, False)
#     cv2.fillPoly(mask, [hv], 255)


image_size = (400, 400)
grid_size = image_size[0] // hradius // 2, int(image_size[1] // hradius * 0.65)
print(grid_size)
hcenters = pack_hexagons(grid_size, hradius, hangle)


# mask = np.zeros(image_size, dtype=np.uint8)
# for i, hc in enumerate(hcenters):
#     mask = place_hexagon_mask(mask, hc, hradius, hangle, i + 1)


# i = 0
# hc = hcenters[i]
# mask = place_hexagon_mask(mask, hc, hradius, hangle, i + 1)

# determine centers of middle index of hcenters


plt.figure()
plt.subplot(121)

plt.scatter(*zip(*hcenters))
# plot box with size image_size
plt.plot(
    [0, image_size[1], image_size[1], 0, 0], [0, 0, image_size[0], image_size[0], 0]
)

colors = plt.cm.viridis(np.linspace(0, 1, len(hcenters)))
for hc, c in zip(hcenters, colors):
    hv = place_hexagon(hc, hradius, hangle, False)
    plt.plot(*zip(*hv), color=c)


# for i, hc in enumerate(hcenters):
# hv = place_hexagon(hc, hradius, hangle, False)
# plt.plot(*zip(*hv), marker="o", color=colors[i])
plt.gca().set_aspect("equal", adjustable="box")

plt.subplot(122)
plt.imshow(mask, cmap="gray")

plt.show()

# %% Try to pattern an image with the hexagon mask
image_size = (400, 400)
output_size = (1200, 1200)
# output_size = image_size
hex_radius = 100
hex_angle = 0.5


# create an image with 2d sine wavees

x = np.linspace(-1, 1, image_size[1])
y = np.linspace(-1, 1, image_size[0])
X, Y = np.meshgrid(x, y)
# img = np.sin(X) * np.cos(Y) ** 2 + np.cos(X) * np.sin(Y)
img = np.sqrt(X**2 + Y**2)

img = img.astype(np.float32)

start_time = time.time()
output_image = pattern.pattern_hex(img, hex_radius, hex_angle, output_size)
end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")

plot_mapping((img, output_image), ncols=2, colorbar=True)


# %%

print(img.dtype)
print(output_image.dtype)


# plt.imshow()

# plot_mapping((img, copy_mask, output_image, target_mask), ncols=2, colorbar=True)


plot_mapping((img, copy_mask, output_image), ncols=3, colorbar=True)

# plt.subplot(121)
# plt.scatter(*zip(*hcenters))

# plt.subplot(122)
# plt.imshow(copy_mask, cmap="gray")


# %%


height, width = image_input.shape[:2]
height, width = (400, 400)

# Example usage
mask = create_hexagon((height, width), 100, fill_value=1)

mask2 = np.zeros((height, width), dtype=np.uint8)
# points = np.array([[[0, 0], [width // 2, 0], [width // 4, height // 2 + 50]]])
# points = np.array(
#     [[[0, 0], [width // 2, 0], [width // 4, height // 2 + 50], [300, 300]]]
# )
# print(points)
# cv2.fillPoly(mask2, points, 255)


plot_mapping((mask, mask2), ncols=2, colorbar=True)

# plt.imshow(mask, cmap="gray")
# plt.gca().set_aspect("equal", adjustable="box")
# plt.show()


# mask = hexagon_mask(mask, (100, 100), 50)


# plt.imshow(mask)
# # plt.colorbar()
# plt.show()


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
