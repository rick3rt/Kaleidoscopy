import cv2
import numpy as np
from kaleidoscope.image import copy_masked_region


def pack_hexagons(grid_size, hex_radius, angle_offset=0):
    """
    Generate a grid of hexagon centers packed in a honeycomb pattern.

    Parameters:
    - grid_size: Tuple (rows, cols) defining the number of hexagons in the grid.
    - hex_radius: Radius of each hexagon.
    - angle_offset: Offset angle in radians for the hexagon orientation.

    Returns:
    - List of tuples [(x1, y1), (x2, y2), ...] representing hexagon centers.
    """
    rows, cols = grid_size
    hex_centers = []

    for r in range(rows):
        for c in range(cols):
            x = c * 1.5 * hex_radius
            y = r * np.sqrt(3) * hex_radius
            if c % 2 == 1:
                y += np.sqrt(3) * hex_radius / 2
            x_rot = x * np.cos(angle_offset) - y * np.sin(angle_offset)
            y_rot = x * np.sin(angle_offset) + y * np.cos(angle_offset)
            hex_centers.append((x_rot, y_rot))

    min_x = cols
    min_y = rows
    vertices = []
    for hc in hex_centers:
        v = place_hexagon(hc, hex_radius, angle_offset)
        min_x = min(np.min(v[:, 0]), min_x)
        min_y = min(np.min(v[:, 1]), min_y)
        vertices.append(v)

    hex_centers = np.array(hex_centers)
    hex_centers = hex_centers - np.array([min_x, min_y])
    hex_centers = np.floor(hex_centers)
    return hex_centers


def place_hexagon(center, hex_radius, angle_offset=0, remove_duplicates=True):
    """
    Determine the corner vertices of a hexagon given its center.

    Parameters:
    - center: Tuple (x, y) representing the center of the hexagon.
    - hex_radius: Radius of the hexagon.

    Returns:
    - Numpy array of shape (6, 2) containing the corner vertices.

    """

    angles = np.linspace(0, 2 * np.pi, 7)
    if remove_duplicates:
        angles = angles[:-1]  # Remove the last point to avoid duplicate
    angles += angle_offset

    vertices = np.array(
        [
            [
                int(center[0] + hex_radius * np.cos(a)),
                int(center[1] + hex_radius * np.sin(a)),
            ]
            for a in angles
        ],
        np.int32,
    )

    return vertices


def place_hexagon_mask(mask, center, hex_radius, angle_offset=0, fill_value=255):
    """
    Create a mask of a hexagon given its center.

    Parameters:
    - mask: Numpy array of shape (h, w) to draw the hexagon on.
    - center: Tuple (x, y) representing the center of the hexagon. If None, the center of the mask is used.
    - hex_radius: Radius of the hexagon.
    - angle_offset: Offset angle in radians for the hexagon orientation.
    - fill_value: Value to fill inside the hexagon.

    Returns:
    - Numpy array of shape (h, w) with the hexagon drawn on
    """

    center_0 = (0, 0)
    center_1 = (mask.shape[1] // 2, mask.shape[0] // 2)
    vertices = place_hexagon(center_0, hex_radius, angle_offset)
    vertices = np.round(vertices).astype(np.int32) + center_1

    tmp_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.fillPoly(tmp_mask, [vertices], fill_value)

    # now circshift the mask to the center
    if center is not None:
        center_offset = np.round(center - center_1).astype(np.int32)
        tmp_mask = np.roll(tmp_mask, center_offset, axis=(1, 0))
    mask = np.maximum(mask, tmp_mask)
    return mask

    # shift vertices by center offset from center_0
    # if center is not None:
    #     center_offset = np.round(center - center_0).astype(np.int32)
    #     vertices = vertices + center_offset
    # return mask


def pattern_hex(source_image, hex_radius, hex_angle, output_size):

    image_size = source_image.shape[:2]
    # grid_size = output_size[0] // hex_radius, output_size[1] // hex_radius
    grid_size = output_size[0] // hex_radius // 2, int(
        output_size[1] // hex_radius * 0.65
    )
    hcenters = pack_hexagons(grid_size, hex_radius, hex_angle)

    copy_mask = np.zeros(image_size, dtype=np.uint8)
    copy_mask = place_hexagon_mask(copy_mask, None, hex_radius, hex_angle)

    output_size2 = np.append(output_size, 3)

    print(output_size2)
    output_image = np.zeros(output_size2, dtype=source_image.dtype)

    target_mask = np.zeros(output_size, dtype=np.uint8)
    for hc in hcenters:
        target_mask = place_hexagon_mask(target_mask, hc, hex_radius, hex_angle)

        # print("source image", source_image.shape)
        # print("copy mask", copy_mask.shape)
        # print("output image", output_image.shape)
        # print("target mask", target_mask.shape)

        output_image = copy_masked_region(
            source_image, copy_mask, output_image, target_mask
        )
        target_mask.fill(0)

    return output_image
