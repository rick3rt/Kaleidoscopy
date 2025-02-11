# %% imports
import cv2
import numpy as np
import os
import imageio
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from myutil import plot_point, plot_vec, show_img_grid, write_video, create_gif

# %% LOad image
image_folder = "image"
output_folder = "output"

filename = "image1.png"
# filename = "image_doppler.png"
# filename = "test_image.png"

# load image
img_input = cv2.imread(os.path.join(image_folder, filename))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
# img_input = img_crop_center(img_input)

plt.imshow(img_input)

# %% Try to make kaleidoscope

h, w = img_input.shape[:2]

num_repeats = 1
angle_range = 2 * np.pi / num_repeats
angle_offset = 0  # np.pi / 8.0
center_out = (w // 2, h // 2)
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


# map theta to range
t = np.linspace(0, 2 * np.pi, 100)
t2 = np.abs((t - 0.5) % angle_range - angle_range / 2)

# plt.figure()
# plt.plot(t, t)
# plt.plot(t, t2)
# plt.plot(t, 0 * t2, "k--")
# plt.title("Angle mapping")


pix_theta_k = np.abs((pix_theta - angle_offset) % angle_range - angle_range / 2)

# convert to cartesian sample points in input image, offset by c_in
Xk = (pix_mag * np.cos(pix_theta_k) + center_in[0]).astype(np.int64)
Yk = (pix_mag * np.sin(pix_theta_k) + center_in[1]).astype(np.int64)
inds_to_remove = (Yk < 0) | (Yk >= h) | (Xk < 0) | (Xk >= w)
Xk[inds_to_remove] = 0
Yk[inds_to_remove] = 0


# show in subplot
# plt.figure()
# plt.subplot(121)
# plt.imshow(pix_theta)
# plt.title("Magnitude")
# plot_point(center_in, "ro")

# plt.subplot(122)
# plt.imshow(pix_theta_k)
# plt.title("Angle")
# plot_point(center_in, "ro")

# %% Apply transformation

img_out = img_input.copy()
tmp = img_out[0, 0].copy()
img_out[0, 0] = (0, 0, 0)
img_out = img_out[Yk, Xk]
img_out[0, 0] = tmp

plt.imshow(img_out)
plot_point(center_out, "ro")


# %%


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


# %%

img_kal = kaleidoscope(img_input, num_repeats=4)

show_img_grid({"Original": img_input, "Kaleidoscope": img_kal})


# %% more

images = {
    "Original": img_input,
    "blank": 0 * img_input + 255,
    "k1": kaleidoscope(img_input, num_repeats=1),
    "k2": kaleidoscope(img_input, num_repeats=2),
    "k3": kaleidoscope(img_input, num_repeats=3),
    "k4": kaleidoscope(img_input, num_repeats=4),
}

show_img_grid(images)


# %% write images to animated gif


# filename = os.path.join(output_folder, "kaleidoscope.gif")
# create_gif(filename, list(images.values()), fps=10)


# %% write images to animated gif

repeats = np.arange(1, 20, 0.25)
repeats = np.concatenate([repeats, repeats[::-1]])

x_offset = np.linspace(0, 50, len(repeats))


images = []
for r in repeats:
    center = (w // 2 + int(x_offset[len(images)]), h // 2)
    img_kal = kaleidoscope(img_input, num_repeats=r, center_in=center)
    img_kal = cv2.cvtColor(img_kal, cv2.COLOR_RGB2BGR)

    images.append(img_kal)

filename = os.path.join(output_folder, "kaleidoscope_varying2.gif")

# create_gif(filename, images, fps=10)
write_video(filename.replace(".gif", ".mp4"), images, fps=10)
print("Done")
