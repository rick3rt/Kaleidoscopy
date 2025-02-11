# %% imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

from util import show_img_grid, img_crop_center, img_rotate_center

# %% function definitions


def kaleidoscope(img, invert=False, angle=0, ntiles=1):
    # # plot input and output images side by side
    imgt = cv2.transpose(img)
    img_h, img_w = img.shape[:2]

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    points = np.array([[[0, 0], [img_w, 0], [img_w, img_h]]])
    cv2.fillPoly(mask, points, 255)
    if invert:
        mask = cv2.bitwise_not(mask)  # optional inverted mask

    # composite img and imgt using mask
    compA = cv2.bitwise_and(imgt, imgt, mask=mask)
    compB = cv2.bitwise_and(img, img, mask=255 - mask)
    comp = cv2.add(compA, compB)

    # rotate composite
    if angle == 90:
        comp = cv2.rotate(comp, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        comp = cv2.rotate(comp, cv2.ROTATE_180)
    elif angle == 270:
        comp = cv2.rotate(comp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        if angle != 0:
            print(
                "Invalid rotation angle. Must be 0, 90, 180, or 270. Using 0 degrees."
            )

    # mirror (flip) horizontally
    mirror = cv2.flip(comp, 1)

    # concatenate horizontally
    top = np.hstack((comp, mirror))

    # mirror (flip) vertically
    bottom = cv2.flip(top, 0)

    # concatenate vertically
    img_out = np.vstack((top, bottom))

    for k in range(ntiles - 1):
        img_out = np.hstack((img_out, img_out))
        img_out = np.vstack((img_out, img_out))

    return img_out


# %% load/show image

image_folder = "image"
output_folder = "output"

filename = "image1.png"
filename = "image_doppler.png"
# filename = "test_image.png"

# load image
img_input = cv2.imread(os.path.join(image_folder, filename))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
img_input = img_crop_center(img_input)

plt.imshow(img_input)
img_to_kal = img_input

# %% rotate image

img_rot = img_rotate_center(img_input, 0, scale=1.0)


show_img_grid(
    {
        "original": img_input,
        "rotated": img_rot,
    }
)

img_to_kal = img_rot
# %%
ntiles = 1
# create kaleidoscope images
images = {
    "original": img_input,
    "to kal": img_to_kal,
    "K0-000d": kaleidoscope(img_to_kal, invert=False, angle=0, ntiles=ntiles),
    "K0-090d": kaleidoscope(img_to_kal, invert=False, angle=90, ntiles=ntiles),
    "K0-180d": kaleidoscope(img_to_kal, invert=False, angle=180, ntiles=ntiles),
    "K0-270d": kaleidoscope(img_to_kal, invert=False, angle=270, ntiles=ntiles),
    "K1-000d": kaleidoscope(img_to_kal, invert=True, angle=0, ntiles=ntiles),
    "K1-090d": kaleidoscope(img_to_kal, invert=True, angle=90, ntiles=ntiles),
    "K1-180d": kaleidoscope(img_to_kal, invert=True, angle=180, ntiles=ntiles),
    "K1-270d": kaleidoscope(img_to_kal, invert=True, angle=270, ntiles=ntiles),
}

# show images
fig = show_img_grid(images, ncols=2, fig_size=(10, 15))

# %% save plotted image fig

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# filename = f"{filename.split('.')[0]}_kaleidoscope.png"


fig.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches="tight")


# %%


# Load the image
# image = cv2.imread('your_image.jpg')
image = img_input

# pad image with zeros, so its double the size
height, width = image.shape[:2]
padh = height // 4
padw = width // 4

image = cv2.copyMakeBorder(
    image, padh, padh, padw, padw, cv2.BORDER_CONSTANT, value=[0, 0, 0]
)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for displaying with matplotlib

# Get the dimensions of the image
height, width = image.shape[:2]
center = (width // 2, height // 2)


# Convert the image from Cartesian to Polar coordinates
polar_image = cv2.warpPolar(
    image, (width, height), center, min(center), cv2.WARP_FILL_OUTLIERS
)

# Apply a circular warp by rotating in polar space
# For example, shift the image horizontally (which corresponds to rotating in polar)
shift_pixels = 200  # You can adjust this value for more/less rotation
polar_image_shifted = np.roll(polar_image, shift_pixels, axis=1)

# Convert back from Polar to Cartesian coordinates
warped_image = cv2.warpPolar(
    polar_image_shifted, (width, height), center, min(center), cv2.WARP_INVERSE_MAP
)

# Display the original and warped images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Warped Circular Image")
plt.imshow(warped_image)
plt.axis("off")

plt.show()


# %%


def hypot(shape):
    return np.sqrt(shape[0] ** 2 + shape[1] ** 2)


def warpMatrix(sz, theta, phi, gamma, scale, fovy):

    st = np.sin(np.radians(theta))
    ct = np.cos(np.radians(theta))
    sp = np.sin(np.radians(phi))
    cp = np.cos(np.radians(phi))
    sg = np.sin(np.radians(gamma))
    cg = np.cos(np.radians(gamma))

    halfFovy = fovy * 0.5
    d = hypot(sz)
    sideLength = scale * d / np.cos(np.radians(halfFovy))
    h = d / (2.0 * np.sin(np.radians(halfFovy)))
    n = h - (d / 2.0)
    f = h + (d / 2.0)

    F = np.zeros((4, 4))  # 4x4 transformation matrix F

    Rtheta = np.eye(4)  # 4x4 rotation matrix around Z-axis by theta degrees
    Rphi = np.eye(4)  # 4x4 rotation matrix around X-axis by phi degrees
    Rgamma = np.eye(4)  # 4x4 rotation matrix around Y-axis by gamma degrees
    T = np.eye(4)  # 4x4 translation matrix along Z-axis by -h units
    P = np.zeros((4, 4))  # Allocate 4x4 projection matrix

    # Rtheta
    Rtheta[0, 0] = Rtheta[1, 1] = ct
    Rtheta[0, 1] = -st
    Rtheta[1, 0] = st

    # Rphi
    Rphi[1, 1] = Rphi[2, 2] = cp
    Rphi[1, 2] = -sp
    Rphi[2, 1] = sp

    # Rgamma
    Rgamma[0, 0] = Rgamma[2, 2] = cg
    Rgamma[0, 2] = -sg
    Rgamma[2, 0] = sg

    # T
    T[2, 3] = -h

    # P
    P[0, 0] = P[1, 1] = 1.0 / np.tan(np.radians(halfFovy))
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -(2.0 * f * n) / (f - n)
    P[3, 2] = -1.0

    # Compose transformations
    F = np.dot(np.dot(np.dot(np.dot(P, T), Rphi), Rtheta), Rgamma)

    # Transform 4x4 points
    halfW = sz[1] / 2
    halfH = sz[0] / 2

    ptsIn = np.array(
        [-halfW, halfH, 0, halfW, halfH, 0, halfW, -halfH, 0, -halfW, -halfH, 0]
    )

    ptsInMat = np.reshape(ptsIn, (4, 1, 3))

    ptsOutMat = cv2.perspectiveTransform(ptsInMat, F)  # Transform points

    ptsInPt2f = np.zeros((4, 2)).astype("float32")
    ptsOutPt2f = np.zeros((4, 2)).astype("float32")

    for i in range(4):
        ptsInPt2f[i] = ptsInMat[i, 0, :2] + np.array([halfW, halfH])
        ptsOutPt2f[i] = (ptsOutMat[i, 0, :2] + np.ones(2)) * (sideLength * 0.5)

    return cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f)


def warpImage(src, theta, phi, gamma, scale, fovy):

    halfFovy = fovy * 0.5
    d = hypot(src.shape)
    sideLength = int(scale * d / np.cos(np.radians(halfFovy)))

    # Compute warp matrix
    M = warpMatrix(src.shape, theta, phi, gamma, scale, fovy)

    # Do actual image war0
    return cv2.warpPerspective(src, M, (sideLength, sideLength))


theta = 5
phi = 30
gamma = 20
scale = 2
fov = 50
img_warp = warpImage(img_input, theta, phi, gamma, scale, fov)

show_img_grid(
    {
        "original": img_input,
        "warped": img_warp,
    }
)
