import cv2
import numpy as np


class Image:

    def load(filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def save(filename, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)

    def pad(image, pad_size):
        h, w = image.shape[:2]

        pad_h = (pad_size[0] - h) // 2
        pad_w = (pad_size[1] - w) // 2

        image = cv2.copyMakeBorder(
            image,
            pad_h,
            pad_h,
            pad_w,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        return image

    def crop(image, crop_size):
        """
        Crop the image to the specified size, around the center.

        Parameters:
        - image: Numpy array of shape (h, w, c).
        - crop_size: Tuple of (crop_h, crop_w).

        Returns:
        - Numpy array of shape (crop_h, crop_w, c).
        """
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)

        crop_y = (h - crop_h) // 2
        crop_x = (w - crop_w) // 2

        image = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        return image

    def resize(image, size):
        return cv2.resize(image, size)

    def scale(image, scale):
        # check if scale is scalar or tuple
        if isinstance(scale, (int, float)):
            return cv2.resize(image, (0, 0), fx=scale, fy=scale)
        else:
            return cv2.resize(image, (0, 0), fx=scale[0], fy=scale[1])


# test Image
# filename = "image/image_doppler.png"
# image = Image.load(filename)

# image2 = Image.pad(image, (1000, 1000))
# image3 = Image.crop(image, (400, 400))
# image4 = Image.pad(image3, (1000, 1000))
# image5 = Image.resize(image3, (1000, 500))
# image6 = Image.scale(image3, 3)
# images = {
#     f"original {image.shape}": image,
#     f"padded {image2.shape}": image2,
#     f"cropped {image3.shape}": image3,
#     f"cropped2 {image4.shape}": image4,
#     f"resized {image5.shape}": image5,
#     f"scaled {image6.shape}": image6,
# }

# show_img_grid(images)


def copy_masked_region(src_img, src_mask, dst_img, dst_mask):
    """
    Copy the pixels from src_img to dst_img using corresponding masks.

    Parameters:
    - src_img: Source image (can be any size).
    - src_mask: Binary mask for source (same number of masked pixels as dst_mask).
    - dst_img: Target image (can be any size).
    - dst_mask: Binary mask for target (same number of masked pixels as src_mask).

    Returns:
    - dst_img with source pixels copied over the masked area.
    """

    n_source = np.count_nonzero(src_mask)
    n_target = np.count_nonzero(dst_mask)
    assert (
        n_source == n_target
    ), f"Number of masked pixels must match (src: {n_source} != dst: {n_target})"

    # Get coordinates of non-zero pixels in both masks
    src_coords = np.column_stack(np.where(src_mask > 0))
    tgt_coords = np.column_stack(np.where(dst_mask > 0))

    # Extract the pixel values from the source image
    src_pixels = src_img[src_mask > 0]

    # Sort coordinates for consistency
    src_coords = src_coords[np.lexsort((src_coords[:, 1], src_coords[:, 0]))]
    tgt_coords = tgt_coords[np.lexsort((tgt_coords[:, 1], tgt_coords[:, 0]))]

    # Assign source pixels to target image
    # dst_img_copy = dst_img.copy()
    # dst_img_copy[tgt_coords[:, 0], tgt_coords[:, 1]] = src_pixels
    dst_img[tgt_coords[:, 0], tgt_coords[:, 1]] = src_pixels

    return dst_img
