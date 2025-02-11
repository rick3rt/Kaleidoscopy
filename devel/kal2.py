# %% imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.interpolate import RegularGridInterpolator

from util import show_img_grid, img_crop_center, img_rotate_center


# %% load/show image

image_folder = "image"
output_folder = "output"

filename = "image3.png"
# filename = "image_doppler.png"
# filename = "test_image.png"

# load image
img_input = cv2.imread(os.path.join(image_folder, filename))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
# img_input = img_crop_center(img_input)

plt.imshow(img_input)


# %%


def rotate_vector_2d(vector, angle_degrees):
    theta = np.radians(angle_degrees)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    v = np.asarray(vector, dtype=np.float64)
    rotated_vector = rotation_matrix @ v
    return rotated_vector


def find_intersection_with_square(point, direction, width, height):
    px, py = point
    dx, dy = direction

    intersections = []

    # Avoid division by zero
    if dx != 0:
        t_left = (0 - px) / dx
        y_left = py + t_left * dy
        if 0 <= y_left <= height and t_left >= 0:
            intersections.append((t_left, (0, y_left)))

        t_right = (width - px) / dx
        y_right = py + t_right * dy
        if 0 <= y_right <= height and t_right >= 0:
            intersections.append((t_right, (width, y_right)))

    if dy != 0:
        t_top = (0 - py) / dy
        x_top = px + t_top * dx
        if 0 <= x_top <= width and t_top >= 0:
            intersections.append((t_top, (x_top, 0)))

        t_bottom = (height - py) / dy
        x_bottom = px + t_bottom * dx
        if 0 <= x_bottom <= width and t_bottom >= 0:
            intersections.append((t_bottom, (x_bottom, height)))

    if not intersections:
        return None  # No valid intersection in the direction of the vector

    # Find the intersection with the smallest positive t (the one in the direction of the vector)
    intersections.sort(key=lambda x: x[0])
    return intersections[0][1]


def triangle_mask(img, tip_location, direction, angle_deg):
    h, w = img.shape[:2]

    # normalize direction
    direction = direction / np.linalg.norm(direction)

    d_l = rotate_vector_2d(direction, -angle_deg / 2)
    d_r = rotate_vector_2d(direction, angle_deg / 2)
    p_l = find_intersection_with_square(tip_location, d_l, w, h)
    p_r = find_intersection_with_square(tip_location, d_r, w, h)

    # determine distance from tip to intersection points
    dist_l = np.linalg.norm(np.array(p_l) - np.array(tip_location))
    dist_r = np.linalg.norm(np.array(p_r) - np.array(tip_location))
    dist = min(dist_l, dist_r)

    p_l = tip_location + dist * d_l
    p_r = tip_location + dist * d_r

    # Create mask
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mask = np.zeros((h, w), dtype=np.bool)

    def is_inside(x, y, triangle):
        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]

        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
        c = 1 - a - b
        return (a >= 0) & (b >= 0) & (c >= 0)

    triangle = np.array([tip_location, p_l, p_r])
    mask = is_inside(X, Y, triangle)
    mask = mask.astype(np.uint8)
    return mask, p_l, p_r


# %% plot functions


def plot_point(p, *args, **kwargs):
    plt.plot(p[0], p[1], *args, **kwargs)


def plot_vec(start, end, *args, **kwargs):
    plt.plot([start[0], end[0]], [start[1], end[1]], *args, **kwargs)


# %% make slice mask triangle
h, w = img_input.shape[:2]

tip = (w // 2, 1)
dir = np.array([1, 1])
angle = 30

msk, line1, line2 = triangle_mask(img_input, tip, dir, angle)

img_masked = cv2.bitwise_and(img_input, img_input, mask=msk)

plt.imshow(img_masked)
plt.plot([tip[0], line1[0]], [tip[1], line1[1]], "r")
plt.plot([tip[0], line2[0]], [tip[1], line2[1]], "r")

# %% MIRROR

img_mirror1 = mirror_image(img_input, tip, line1)
msk_mirror1 = mirror_image(msk.astype(np.float64) * 255.0, tip, line1)
msk_mirror1 = np.clip(msk_mirror1, 0, 1).astype(np.uint8)

msk_mirror1 = msk_mirror1 - (msk_mirror1 & msk)

plt.imshow(msk_mirror1)


img_mirror1 = cv2.bitwise_and(img_mirror1, img_mirror1, mask=msk_mirror1)

plt.imshow(img_mirror1)
plt.plot([tip[0], line1[0]], [tip[1], line1[1]], "r")
plt.plot([tip[0], line2[0]], [tip[1], line2[1]], "r")
plt.show()

img_merge = cv2.add(img_masked, img_mirror1)

plt.imshow(img_merge)
plt.plot([tip[0], line1[0]], [tip[1], line1[1]], "r")
plt.plot([tip[0], line2[0]], [tip[1], line2[1]], "r")


# %% Extract image with mask

msk = msk.astype(np.uint8)
img_extract = cv2.bitwise_and(img_input, img_input, mask=msk)

plt.imshow(img_extract)


# %% make slice mask triangle


ARROW_LEN = 50
DIRECTION = np.array([1, 1])
DIRECTION = DIRECTION / np.linalg.norm(DIRECTION)
ANGLE = 90

h, w = img_input.shape[:2]

triangle_tip = np.array([50, 50])

d_l = rotate_vector_2d(DIRECTION, -ANGLE / 2)
d_r = rotate_vector_2d(DIRECTION, ANGLE / 2)
p_l = find_intersection_with_square(triangle_tip, d_l, w, h)
p_r = find_intersection_with_square(triangle_tip, d_r, w, h)

# determine distance from tip to intersection points
dist_l = np.linalg.norm(np.array(p_l) - np.array(triangle_tip))
dist_r = np.linalg.norm(np.array(p_r) - np.array(triangle_tip))
dist = min(dist_l, dist_r)


# msk = create_triangle_mask(w, h, triangle_tip, DIRECTION, 60)
msk = triangle_mask(img_input, triangle_tip, DIRECTION, ANGLE)


plt.imshow(msk)
plot_point(triangle_tip, "ro")
plot_vec(triangle_tip, triangle_tip + ARROW_LEN * DIRECTION, "r")
plot_vec(triangle_tip, triangle_tip + dist * d_l, "b", linewidth=2)
plot_vec(triangle_tip, triangle_tip + dist * d_r, "b", linewidth=2)
plt.autoscale(False)

plot_point(p_l, "ro")
plot_point(p_r, "ro")

# plt.plot(x, yf1, "b")
# plt.plot(x, yf2, "b")


# plt.imshow(mask)
# plt.plot(triangle_tip[0], triangle_tip[1], "ro")


# show_img_grid({
#     "original": img_input,
#     "mask": mask
# })


# %%

v = np.array([1, 1])
v = v / np.linalg.norm(v)
print(v)
msk = create_triangle_mask(400, 400, triangle_tip, v, 60)


plt.imshow(msk)
plot_point(triangle_tip, "ro")
plot_vec(triangle_tip, triangle_tip + ARROW_LEN * v, "r", linewidth=2)
# %% TEST MIRRORING IMAGE


def mirror_image(img, p1, p2):
    w, h = img.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    [Y, X] = np.meshgrid(y, x)

    p1 = np.flip(p1)  # DONT ASK
    p2 = np.flip(p2)

    img = img.astype(np.float64)
    interp = RegularGridInterpolator((x, y), img, bounds_error=False, fill_value=0)

    def mirror(Xv, Yv, P, R):
        vx, vy = R[0] - P[0], R[1] - P[1]
        x, y = P[0] - Xv, P[1] - Yv
        r = 1.0 / (vx * vx + vy * vy)
        return (
            Xv + 2.0 * (x - x * vx * vx * r - y * vx * vy * r),
            Yv + 2.0 * (y - y * vy * vy * r - x * vx * vy * r),
        )

    Xm, Ym = mirror(X, Y, p1, p2)  # mirror interpolation grid
    img_mirror = interp((Xm, Ym))  # interpolate image

    img_mirror = np.clip(img_mirror, 0, 255)  # clip to valid color range
    img_mirror = img_mirror.astype(np.uint8)  # convert to uint8
    return img_mirror


img = img_input
h, w = img.shape[:2]

p1 = np.array([w // 2, 0], dtype=np.float64)
p2 = np.array([w // 4, h - 1], dtype=np.float64)
img_mirror = mirror_image(img, p1, p2)

print(img.shape)
print(img_mirror.shape)

plt.figure(figsize=(20, 10))
plt.subplot(211)
plt.imshow(img)
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r")

plt.subplot(212)
plt.imshow(img_mirror)
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r")


# %%


def core(img, N, out, r_start, r_out, c_in, c_out, scale):
    in_rows, in_cols = img.shape[:2]
    if c_in is None:
        c_in = (dim // 2 for dim in (in_rows, in_cols))
    c_y, c_x = c_in

    r_start %= 2 * np.pi
    width = np.pi / N
    r_end = r_start + width

    if out == "same":
        out = np.empty((in_rows, in_cols, 3), dtype=np.uint8)
    elif out == "full":
        quarter = np.pi / 2
        r_mid = (r_start + r_end) / 2
        if 0 <= r_mid < quarter:
            dy = in_rows - c_y
            dx = in_cols - c_x
        elif quarter <= r_mid <= 2 * quarter:
            dy = in_rows - c_y
            dx = c_x
        elif 2 * quarter <= r_mid <= 3 * quarter:
            dy = c_y
            dx = c_x
        else:
            dy = c_y
            dx = in_cols - c_x
        s = int(np.ceil(2 * np.sqrt(dx * dx + dy * dy) * scale))
        out = np.empty((s, s, 3), dtype=np.uint8)
    else:
        out = np.empty((out, out, 3), dtype=np.uint8)

    out_rows, out_cols = out.shape[:2]
    if c_out is None:
        c_out = (dim // 2 for dim in (out_rows, out_cols))
    co_y, co_x = c_out

    # create sample points and offset to center of output image
    Xp, Yp = np.meshgrid(np.arange(out_cols), np.arange(out_rows))
    Xp -= co_x
    Yp -= co_y

    # calculate magnitude and angle of each sample point in input image
    mag_p = np.sqrt(Xp * Xp + Yp * Yp) / scale
    theta_p = np.abs(((np.arctan2(Xp, Yp) - r_out) % (2 * width)) - width) + r_start

    # convert to cartesian sample points in input image, offset by c_in
    Y = (mag_p * np.sin(theta_p) + c_y).astype(np.int64)
    X = (mag_p * np.cos(theta_p) + c_x).astype(np.int64)

    # set outside valid region pixels to black (avoid index error)
    # temporarily use pixel [0,0] of input image
    old = img[0, 0].copy()
    img[0, 0] = (0, 0, 0)

    bad = (Y < 0) | (Y >= in_rows) | (X < 0) | (X >= in_cols)
    Y[bad] = 0
    X[bad] = 0

    # sample input image to set each pixel of out
    out[:] = img[Y, X]

    img[0, 0] = old  # restore input [0,0] to its initial value
    return out, c_x, c_y, r_start, r_end


def kaleido(
    img,
    N=10,
    out="same",
    r_start=0,
    r_out=0,
    c_in=None,
    c_out=None,
    scale=1,
):
    """Return a kaleidoscope from img, with specified parameters.

    'img' is a 3-channel uint8 numpy array of image pixels.
    'N' is the number of mirrors.
    'out' can be 'same', 'full', or a 3-channel uint8 array to fill.
    'r_start' is the selection rotation from the input image [clock radians].
    'r_out' is the rotation of the output image result [clock radians].
    'c_in' is the origin point of the sample sector from the input image.
        If None defaults to the center of the input image [c_y,c_x].
    'c_out' is the center of the kaleidoscope in the output image. If None
        defaults to the center point of the output image [c_y, c_x].
    'scale' is the scale of the output kaleidoscope. Default 1.
    'annotate' is a boolean denoting whether to annotate the input image to
        display the selected region. Default True.

    """
    out, c_x, c_y, r_start, r_end = core(
        img, N, out, r_start, r_out, c_in, c_out, scale
    )
    return out


c_in = None  #  (0, 0)
c_out = None
r_start = np.radians(0)
r_out = np.radians(0)


img_kal = kaleido(
    img_input, N=4, r_start=r_start, r_out=r_out, c_in=c_in, c_out=None, scale=1
)


plt.figure()
plt.subplot(121)
plt.imshow(img_input)
if not c_in:
    c_in = (img_input.shape[1] // 2, img_input.shape[0] // 2)
plot_point(c_in, "ro")

plt.subplot(122)
plt.imshow(img_kal)

if not c_out:
    c_out = (img_kal.shape[1] // 2, img_kal.shape[0] // 2)

plot_point(c_out, "ro")


# show_img_grid({"original": img_input, "kaleidoscope": img_kal})


# %%


# 2D Points P=[x,y] and R=[x,y] are arbitrary points on line,
# Q=[x,y] is point for which we want to find reflection
# returns solution vector [x,y]
def mirror(Q, P, R):
    vx, vy = R[0] - P[0], R[1] - P[1]
    x, y = P[0] - Q[0], P[1] - Q[1]
    r = 1 / (vx * vx + vy * vy)
    return Q[0] + 2 * (x - x * vx * vx * r - y * vx * vy * r), Q[1] + 2 * (
        y - y * vy * vy * r - x * vx * vy * r
    )


P = (0, 0)
R = (0, 1)
Q = (0.5, 0.5)
Q_reflected = mirror(Q, P, R)

# %%

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


def mirror2(Xv, Yv, P, R):
    vx, vy = R[0] - P[0], R[1] - P[1]
    x, y = P[0] - Xv, P[1] - Yv
    r = 1 / (vx * vx + vy * vy)
    return (
        Xv + 2 * (x - x * vx * vx * r - y * vx * vy * r),
        Yv + 2 * (y - y * vy * vy * r - x * vx * vy * r),
    )


# row, col
w, h = (200, 200)
# Q = np.array([4, 1], dtype=np.float64)
P = np.array([w // 2, 0], dtype=np.float64)
R = np.array([w // 4, h - 1], dtype=np.float64)


x = np.arange(w)
y = np.arange(h)
[X, Y] = np.meshgrid(x, y)


img = np.zeros((w, h))
# img = img * 40
# img[0,0]=0
img[30, 30] = 255

img = gaussian_filter(img, sigma=1)

# gaussian blur img
# img = cv2.GaussianBlur(img, (5, 5), 0)


# print(x.shape)
# print(y.shape)
# print(img.shape)

interp = RegularGridInterpolator((x, y), img, bounds_error=False, fill_value=0)


Q_reflected = mirror(Q, P, R)
Xm, Ym = mirror2(X, Y, P, R)
img2 = interp((Xm, Ym))

# clip to [0, 255]
img2 = np.clip(img2, 0, 255)

print(img.shape)
print(img2.shape)


plt.subplot(121)
plt.imshow(img.T, cmap="gray")
plt.plot([P[0], R[0]], [P[1], R[1]], "r")
# plt.plot(*P, "ro")
# plt.plot(*R, "ro")
# plt.plot(*Q, "bo")
# plt.plot(*Q_reflected, "bo")

plt.subplot(122)
plt.imshow(img2, cmap="gray")
plt.plot([P[0], R[0]], [P[1], R[1]], "r")
# plt.plot(*P, "ro")
# plt.plot(*R, "ro")
# plt.plot(*Q, "bo")
# plt.plot(*Q_reflected, "bo")


# plt.grid(True, xticks=np.arange(w), yticks=np.arange(h))
# plt.plot(*Q, "bo")
# plt.plot(*Q_reflected, "bo")


print(Xm)
# %%

x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])


def ff(x, y):
    return x**2 + y**2


xg, yg = np.meshgrid(x, y, indexing="ij")
data = ff(xg, yg)
interp = RegularGridInterpolator((x, y), data, bounds_error=False, fill_value=None)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(xg.ravel(), yg.ravel(), data.ravel(), s=60, c="k", label="data")


xx = np.linspace(-4, 9, 31)
yy = np.linspace(-4, 9, 31)
X, Y = np.meshgrid(xx, yy, indexing="ij")

# interpolator
ax.plot_wireframe(
    X,
    Y,
    interp((X, Y)),
    rstride=3,
    cstride=3,
    alpha=0.4,
    color="m",
    label="linear interp",
)

# ground truth
ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3, alpha=0.4, label="ground truth")
plt.legend()
plt.show()
