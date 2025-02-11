import math
import imageio
import cv2
import numpy as np
from matplotlib import pyplot as plt


def img_rect_crop(img):
    # crop largest rectangular image from the center
    h, w = img.shape[:2]
    min_dim = min(h, w)
    img = img[
        (h - min_dim) // 2 : (h + min_dim) // 2, (w - min_dim) // 2 : (w + min_dim) // 2
    ]
    return img


def img_center_crop(image, crop_width, crop_height):
    """
    Crops the image around its center to the specified width and height.

    Parameters:
        image (numpy.ndarray): Input image.
        crop_width (int): Desired width of the cropped image.
        crop_height (int): Desired height of the cropped image.

    Returns:
        numpy.ndarray: Cropped image.
    """
    # Get the original dimensions
    height, width = image.shape[:2]

    # Ensure the crop size doesn't exceed the image size
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    # Calculate the coordinates for cropping
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    end_x = start_x + crop_width
    end_y = start_y + crop_height

    # Crop and return the image
    return image[start_y:end_y, start_x:end_x]


def img_rotate_center(image, angle, center=None, scale=1.0, scale_fit=True):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # if scale_fit:
    #     # determine size of black areas and scale to not have black areas
    #     # calculate rotation matrix
    #     M = cv2.getRotationMatrix2D(center, angle, scale)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def map(value, from_min, from_max, to_min, to_max):

    from_range = from_max - from_min
    to_range = to_max - to_min

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - from_min) / from_range

    # Convert the 0-1 range into a value in the right range.
    return to_min + (valueScaled * to_range)
