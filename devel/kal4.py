# %% imports
import cv2
import numpy as np
import os
import imageio
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from util import plot_point, plot_vec, show_img_grid, show_img, img_crop_center

# %% LOad image
image_folder = "image"
output_folder = "output"

filename = "image1.png"
filename = "image_doppler.png"
# filename = "test_image.png"

# load image
img_input = cv2.imread(os.path.join(image_folder, filename))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
# img_input = img_crop_center(img_input)


# add padding around image
pad = 100
img_input = cv2.copyMakeBorder(
    img_input, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
)

plt.imshow(img_input)

# %% functions


def kaleidoscope(
    img,
    num_repeats=1,
    angle_offset_in=0,
    angle_offset_out=0,
    center_in=None,
    center_out=None,
):

    h, w = img.shape[:2]

    if not center_out:
        center_out = (w // 2, h // 2)
    if not center_in:
        center_in = (w // 2, h // 2)
    angle_range = 2 * np.pi / num_repeats

    x = np.arange(w)
    y = np.arange(h)

    # create meshgrid
    x -= center_out[0]
    y -= center_out[1]
    X, Y = np.meshgrid(x, y)

    # calculate magnitude and angle of each sample point in input image
    pix_mag = np.sqrt(X**2 + Y**2)
    pix_theta = np.arctan2(X, Y) + angle_offset_in

    pix_theta_k = np.abs((pix_theta - angle_offset_out) % angle_range - angle_range / 2)

    # convert to cartesian sample points in input image, offset by c_in
    Xk = (pix_mag * np.cos(pix_theta_k) + center_in[0]).astype(np.int64)
    Yk = (pix_mag * np.sin(pix_theta_k) + center_in[1]).astype(np.int64)
    inds_to_remove = (Yk < 0) | (Yk >= h) | (Xk < 0) | (Xk >= w)
    Xk[inds_to_remove] = 0
    Yk[inds_to_remove] = 0

    img_out = img.copy()
    tmp = img_out[0, 0].copy()
    img_out[0, 0] = (0, 0, 0)
    img_out = img_out[Yk, Xk]
    img_out[0, 0] = tmp
    return img_out


def map(value, from_min, from_max, to_min, to_max):

    from_range = from_max - from_min
    to_range = to_max - to_min

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - from_min) / from_range

    # Convert the 0-1 range into a value in the right range.
    return to_min + (valueScaled * to_range)


# %% Try to make kaleidoscope

h, w = img_input.shape[:2]

num_repeats = 3
angle_range = 2 * np.pi / num_repeats
angle_offset = 0  # np.pi / 8.0
center_out = (w // 2, h // 2)
center_in = (w // 2, h // 2)
edge_len = 100


x = np.arange(w)
y = np.arange(h)

# create meshgrid
x -= center_out[0]
y -= center_out[1]
X, Y = np.meshgrid(x, y)

# calculate magnitude and angle of each sample point in input image
pix_mag = np.sqrt(X**2 + Y**2)
pix_theta = np.arctan2(X, Y)

# show in subplot
plt.figure()
plt.subplot(121)
plt.imshow(pix_mag)
plt.title("Magnitude")
plot_point(center_in, "ro")

plt.subplot(122)
plt.imshow(pix_theta)
plt.title("Angle")
plot_point(center_in, "ro")


pix_theta_k = np.abs((pix_theta - angle_offset) % angle_range - angle_range / 2)
# pix_theta_k = np.pow(pix_theta_k, 1.5)
# pix_theta_k = pix_theta_k + map(pix_mag, 0, np.max(pix_mag[:]), -1, 0.5)
# pix_theta_k = pix_theta_k % angle_range
# pix_theta_k = pix_theta_k % (2 * np.pi)

# convert to cartesian sample points in input image, offset by c_in
Xk = (pix_mag * np.cos(pix_theta_k) + center_in[0]).astype(np.int64)
Yk = (pix_mag * np.sin(pix_theta_k) + center_in[1]).astype(np.int64)
inds_to_remove = (Yk < 0) | (Yk >= h) | (Xk < 0) | (Xk >= w)
Xk[inds_to_remove] = 0
Yk[inds_to_remove] = 0


# %% Create pattern


def create_pattern(img, num_repeats, angle_offset, center_in, center_out):

    h, w = img.shape[:2]

    if center_out is None:
        center_out = (w // 2, h // 2)
    if center_in is None:
        center_in = (w // 2, h // 2)

    x = np.arange(w)
    y = np.arange(h)

    # create meshgrid
    x -= center_out[0]
    y -= center_out[1]
    X, Y = np.meshgrid(x, y)

    # calculate magnitude and angle of each sample point in input image
    pix_mag = np.sqrt(X**2 + Y**2)
    pix_theta = np.arctan2(X, Y)
    if num_repeats == 0:
        pix_theta_k = pix_theta - angle_offset
    else:
        angle_range = 2 * np.pi / num_repeats
        pix_theta_k = np.abs((pix_theta - angle_offset) % angle_range - angle_range / 2)

    # pix_theta_k = np.pow(pix_theta_k, 1.5)
    # pix_theta_k = pix_theta_k + map(pix_mag, 0, np.max(pix_mag[:]), -1, 0.5)
    # pix_theta_k = pix_theta_k % angle_range
    # pix_theta_k = pix_theta_k % (2 * np.pi)

    return pix_mag, pix_theta_k


def kaleidoscope_pattern(
    img_input, pix_mag, pix_theta_k, center_in=None, invert_sin=False
):

    print("Kaleidoscope pattern")
    print("\timg_input:", img_input.shape)
    print("\tpix_mag:", pix_mag.shape)
    print("\tpix_theta_k:", pix_theta_k.shape)

    h, w = img_input.shape[:2]
    if center_in is None:
        center_in = (w // 2, h // 2)

    # convert to cartesian sample points in input image, offset by c_in
    if invert_sin:
        Xk = (pix_mag * np.sin(pix_theta_k) + center_in[0]).astype(np.int64)
        Yk = (pix_mag * np.cos(pix_theta_k) + center_in[1]).astype(np.int64)
    else:
        Xk = (pix_mag * np.cos(pix_theta_k) + center_in[0]).astype(np.int64)
        Yk = (pix_mag * np.sin(pix_theta_k) + center_in[1]).astype(np.int64)
    inds_to_remove = (Yk < 0) | (Yk >= h) | (Xk < 0) | (Xk >= w)
    Xk[inds_to_remove] = 0
    Yk[inds_to_remove] = 0

    # Apply transformation
    img_out = img_input.copy()
    tmp = img_out[0, 0].copy()
    img_out[0, 0] = (0, 0, 0)
    img_out = img_out[Yk, Xk]
    img_out[0, 0] = tmp

    return img_out


num_repeats = 1
angle_offset_in = 0  # np.pi / 8.0
angle_offset_out = 0  # np.pi / 8.0
center_out = None  #  (w // 2, h // 2)
center_in = None  #  (w // 2, h // 2)
edge_len = 100


# pix_mag, pix_theta_k = create_pattern(
#     img_input, angle_offset, center_in, center_out, 1, edge_len
# )

# create 9 tiled copies of the pattern
# pix_mag = np.tile(pix_mag, (4, 3))
# pix_theta_k = np.tile(pix_theta_k, (1, 3))
# pix_theta_k = np.concatenate([np.flip(pix_theta_k), pix_theta_k], axis=0)
# pix_theta_k = np.concatenate([np.flip(pix_theta_k), pix_theta_k], axis=0)


img_kaleidoscope0 = kaleidoscope(
    img_input, num_repeats, angle_offset_in, angle_offset_out, center_in, center_out
)

pix_mag, pix_theta_k = create_pattern(
    img_input, 4.6, angle_offset, center_in, center_out
)
img_kaleidoscope1 = kaleidoscope_pattern(img_input, pix_mag, pix_theta_k, None, True)

img_kaleidoscope = img_kaleidoscope0

pad = 150
img_kaleidoscope_crop = img_kaleidoscope[pad:-pad, pad:-pad]
pix_mag_crop = pix_mag[pad:-pad, pad:-pad]
pix_theta_k_crop = pix_theta_k[pad:-pad, pad:-pad]


grid_width = 4


def stacker(input, size=(1, 3), axis=0):
    input = np.tile(input, size)
    input = np.concatenate([np.flip(input), input], axis=axis)
    input = np.concatenate([np.flip(input), input], axis=axis)
    return input


# create tiled copies of the pattern
img_kaleidoscope_tmp = np.tile(img_kaleidoscope_crop, (4, grid_width, 1))
# print("img_kaleidoscope_tmp.shape", img_kaleidoscope_tmp.shape)
pix_mag2 = np.tile(pix_mag_crop, (4, grid_width))
# pix_theta_k2 = np.tile(pix_theta_k_crop, (1, 3))
# pix_theta_k2 = np.concatenate([np.flip(pix_theta_k2), pix_theta_k2], axis=0)
# pix_theta_k2 = np.concatenate([np.flip(pix_theta_k2), pix_theta_k2], axis=0)

pix_theta_k2 = stacker(pix_theta_k_crop, size=(1, grid_width), axis=0)

# print("img_kaleidoscope_crop.shape", img_kaleidoscope_crop.shape)
# img_kaleidoscope_tmp = stacker(img_kaleidoscope_crop, (1, 3, 1), 0)
# print("pix_theta_k2.shape", pix_theta_k2.shape)
# print("img_kaleidoscope_tmp.shape", img_kaleidoscope_tmp.shape)


img_kaleidoscope_final = kaleidoscope_pattern(
    img_kaleidoscope_tmp,
    pix_mag2,
    pix_theta_k2,
    # img_kaleidoscope_tmp, pix_mag_crop, pix_theta_k_crop, Xk_crop, Yk_crop
)


# pix_mag, pix_theta_k = create_pattern(img_input, angle_offset, num_repeats, edge_len)
# images = {
#     "pixmag2": pix_mag2,
#     "pix_theta_k2": pix_theta_k2,
# }

# show_img_grid(images)

# print(pix_mag2.shape, pix_theta_k2.shape)
# print(img_kaleidoscope1.shape)


# print(img_kaleidoscope1)


images = {
    "Original": img_input,
    # "img0": img_kaleidoscope0,
    "img1": img_kaleidoscope1,
    "img crop": img_kaleidoscope_crop,
    "tmp": img_kaleidoscope_tmp,
    "output": img_kaleidoscope_final,
}
show_img_grid(images)

# crop image to 1000x1000
pad = 100
img_kaleidoscope_final = img_kaleidoscope_final[pad:-pad, pad:-pad]


show_img(img_kaleidoscope_final)

# %%


# show in subplot
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(pix_mag2)
plt.title("distance")
plt.colorbar(shrink=0.7)


plt.subplot(122)
plt.imshow(pix_theta_k2)
plt.title("Angle")
plt.colorbar(shrink=0.7)
plot_point(center_in, "ro")
# %%


# # show in subplot
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(Xk_crop)
plt.title("Xk crop")
plt.colorbar(shrink=0.7)
# plot_point(center_in, "ro")


plt.subplot(122)
plt.imshow(Yk_crop)
plt.title("Yk crop")
plt.colorbar(shrink=0.7)
# plot_point(center_in, "ro")


# plt.figure()
# plt.imshow(img_out)


# %%


num_repeats = 3
angle_range = 2 * np.pi / num_repeats
angle_offset_in = 0  # np.pi / 8.0
angle_offset_out = 0  # np.pi / 8.0
center_out = (w // 2, h // 2)
center_in = (w // 2, h // 2)
edge_len = 100


img_kaleidoscope = kaleidoscope(
    img_input, num_repeats, angle_offset_in, angle_offset_out, center_in, center_out
)

# crop the image
pad = 150
img_kaleidoscope_crop = img_kaleidoscope[pad:-pad, pad:-pad]


img_kaleidoscope2 = kaleidoscope(
    img_kaleidoscope_crop, 1, angle_offset_in, angle_offset_out
)


images = {
    "Original": img_input,
    "Kaleidoscope": img_kaleidoscope,
    "Kaleidoscope (Cropped)": img_kaleidoscope_crop,
    "img_kaleidoscope2": img_kaleidoscope2,
}
show_img_grid(images)


# %%


# %%

# Apply transformation
img_out = img_input.copy()
tmp = img_out[0, 0].copy()
img_out[0, 0] = (0, 0, 0)
img_out = img_out[Yk, Xk]
img_out[0, 0] = tmp

# %% ==========================================================
# % ==========================================================
# % ==========================================================


def symmetry_corners(center, size, num_repeats):
    """Returns the vertices of a hexagon given its center and size."""
    angles = np.linspace(0, 2 * np.pi, num_repeats * 2 + 1)
    # if num_repeats % 2 == 1:
    #     angles = angles + (angles[1]-angles[0])/2

    x = center[0] + size * np.cos(angles)
    y = center[1] + size * np.sin(angles)
    return x, y


hpoints = symmetry_corners(center_out, edge_len, num_repeats)


plt.figure()
plt.imshow(img_out)
plot_point(center_out, "ro")
plt.plot(hpoints[0], hpoints[1], "r-")


tile_centers, idx_center_tile = generate_tiling_centers("hexagon", 5, 5, edge_len)
tile_centers += center_out

# show in subplot
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(pix_mag)
plt.title("distance")
plt.colorbar(shrink=0.7)
plot_point(center_in, "ro")
plt.scatter(tile_centers[:, 0], tile_centers[:, 1], c="r", s=10)


plt.subplot(122)
plt.imshow(pix_theta_k)
plt.title("Angle")
plt.colorbar(shrink=0.7)
plot_point(center_in, "ro")
plt.plot(hpoints[0], hpoints[1], "r-")
plt.scatter(tile_centers[:, 0], tile_centers[:, 1], c="r", s=10)

for c in tile_centers:
    hp = symmetry_corners(c, edge_len, num_repeats)
    plt.plot(hp[0], hp[1], "b-")


# %%


def generate_tiling_centers(tiling_type, rows, cols, size=1):
    """
    Generates the centers of regular tilings (triangles, squares, hexagons).

    Parameters:
    - tiling_type: 'triangle', 'square', or 'hexagon'
    - rows: Number of rows in the tiling
    - cols: Number of columns in the tiling
    - size: The distance from the center to a vertex (radius)

    Returns:
    - centers: A list of (x, y) coordinates for each tile's center
    """
    centers = []

    if tiling_type == "square":
        for row in range(rows):
            for col in range(cols):
                x = col * 2 * size
                y = row * 2 * size
                centers.append((x, y))

    elif tiling_type == "triangle":
        height = np.sqrt(3) * size  # Height of an equilateral triangle
        for row in range(rows):
            for col in range(cols):
                x = col * size + (row % 2) * (size / 2)
                y = row * height / 2
                centers.append((x, y))

    elif tiling_type == "hexagon":
        height = np.sqrt(3) * size  # Vertical distance between hex centers
        for row in range(rows):
            for col in range(cols):
                x = col * 3 * size + (1.5 * size if row % 2 else 0)
                y = row * height
                centers.append((x, y))

    else:
        raise ValueError(
            "Invalid tiling type. Choose from 'triangle', 'square', or 'hexagon'."
        )

    centers = np.array(centers)

    # move the point closest to the mean of all points to the origin

    mean = np.mean(centers, axis=0)
    dist = np.linalg.norm(centers - mean, axis=1)
    closest = np.argmin(dist)
    centers -= centers[closest]

    return centers, closest


tile_centers, tile_center_idx = generate_tiling_centers("hexagon", 5, 5, 20)
# print(tile_centers)
# print(tile_center_idx)


plt.figure(figsize=(8, 8))
plt.scatter(tile_centers[:, 0], tile_centers[:, 1])
plt.scatter(tile_centers[tile_center_idx, 0], tile_centers[tile_center_idx, 1])
# plt.gca().set_aspect("equal", adjustable="box")


# %%


def regular_polygon(center, radius, num_sides):
    angles = np.linspace(0, 2 * np.pi, num_sides + 1)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return x, y


def plot_octagon_square_tiling(rows, cols, size):
    fig, ax = plt.subplots(figsize=(8, 8))

    for row in range(rows):
        for col in range(cols):
            x_offset = col * (2 * size + size)
            y_offset = row * (2 * size + size)

            # Draw octagon
            oct_x, oct_y = regular_polygon((x_offset, y_offset), size, 8)
            ax.fill(oct_x, oct_y, edgecolor="black", facecolor="lightblue")

            # Draw squares in the gaps
            square_offsets = [
                (x_offset + size, y_offset),
                (x_offset - size, y_offset),
                (x_offset, y_offset + size),
                (x_offset, y_offset - size),
            ]

            for sq_x, sq_y in square_offsets:
                square_x, square_y = regular_polygon((sq_x, sq_y), size / np.sqrt(2), 4)
                ax.fill(square_x, square_y, edgecolor="black", facecolor="lightgreen")

    ax.set_aspect("equal")
    plt.axis("off")
    plt.show()


# Example usage
plot_octagon_square_tiling(5, 5, 20)
