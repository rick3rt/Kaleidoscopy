import math
import imageio
import cv2
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


def plot_point(p, *args, **kwargs):
    plt.plot(p[0], p[1], *args, **kwargs)


def plot_vec(start, end, *args, **kwargs):
    plt.plot([start[0], end[0]], [start[1], end[1]], *args, **kwargs)


def create_gif(output_path, frames, fps=10):
    """
    Creates a GIF from a list of frames.

    Parameters:
    - frames: List of images (NumPy arrays in BGR format from OpenCV).
    - output_path: Path to save the GIF (e.g., 'output.gif').
    - fps: Frames per second for the GIF.
    """
    # Convert frames from BGR (OpenCV) to RGB (for GIF)
    rgb_frames = frames
    # rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Save frames as a GIF
    imageio.mimsave(output_path, rgb_frames, fps=fps)


def write_video(output_path, frames, fps=30):
    """
    Writes a stack of images to an MP4 video file.

    Parameters:
    - frames: List of images (NumPy arrays in BGR format).
    - output_path: Path to save the MP4 file (e.g., 'output.mp4').
    - fps: Frames per second for the video.
    """
    if not frames:
        raise ValueError("No frames provided to write to video.")

    # Get frame dimensions from the first frame
    height, width, channels = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        if frame.shape != (height, width, channels):
            raise ValueError("All frames must have the same dimensions.")
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_path}")
