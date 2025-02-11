import numpy as np


def kaleidoscope(
    img,
    num_repeats=1,
    angle_offset_in=0,
    angle_offset_out=0,
    center_in=None,
    center_out=None,
    invert_sin=False,
):

    pix_mag, pix_theta_k = create_pattern(
        img,
        num_repeats,
        angle_offset_in,
        angle_offset_out,
        center_in,
        center_out,
    )

    return kaleidoscope_pattern(
        img,
        pix_mag,
        pix_theta_k,
        center_in,
        invert_sin,
    )

    # h, w = img.shape[:2]

    # if not center_out:
    #     center_out = (w // 2, h // 2)
    # if not center_in:
    #     center_in = (w // 2, h // 2)

    # x = np.arange(w)
    # y = np.arange(h)

    # # create meshgrid
    # x -= center_out[0]
    # y -= center_out[1]
    # X, Y = np.meshgrid(x, y)

    # # calculate magnitude and angle of each sample point in input image
    # pix_mag = np.sqrt(X**2 + Y**2)
    # pix_theta = np.arctan2(X, Y) + angle_offset_in

    # # pix_theta_k = np.abs((pix_theta - angle_offset_out) % angle_range - angle_range / 2)

    # if num_repeats == 0:
    #     pix_theta_k = pix_theta - angle_offset_in

    # else:
    #     angle_range = 2 * np.pi / num_repeats
    #     pix_theta_k = np.abs(
    #         (pix_theta - angle_offset_in) % angle_range - angle_range / 2
    #     )

    # # convert to cartesian sample points in input image, offset by c_in
    # if invert_sin:
    #     Xk = (pix_mag * np.sin(pix_theta_k) + center_in[0]).astype(np.int64)
    #     Yk = (pix_mag * np.cos(pix_theta_k) + center_in[1]).astype(np.int64)
    # else:
    #     Xk = (pix_mag * np.cos(pix_theta_k) + center_in[0]).astype(np.int64)
    #     Yk = (pix_mag * np.sin(pix_theta_k) + center_in[1]).astype(np.int64)
    # inds_to_remove = (Yk < 0) | (Yk >= h) | (Xk < 0) | (Xk >= w)
    # Xk[inds_to_remove] = 0
    # Yk[inds_to_remove] = 0

    # img_out = img.copy()
    # tmp = img_out[0, 0].copy()
    # img_out[0, 0] = (0, 0, 0)
    # img_out = img_out[Yk, Xk]
    # img_out[0, 0] = tmp
    # return img_out


def create_pattern(
    img,
    num_repeats,
    angle_offset_in,
    angle_offset_out,
    center_in,
    center_out,
):

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
    pix_theta = np.arctan2(X, Y) - angle_offset_in

    if num_repeats == 0:
        pix_theta_k = pix_theta

    else:
        angle_range = 2 * np.pi / num_repeats
        pix_theta_k = np.abs((pix_theta % angle_range) - angle_range / 2)

    pix_theta_k = pix_theta_k + angle_offset_out % 2 * np.pi

    # pix_theta_k = np.pow(pix_theta_k, 1.5)
    # pix_theta_k = pix_theta_k + map(pix_mag, 0, np.max(pix_mag[:]), -1, 0.5)
    # pix_theta_k = pix_theta_k % angle_range
    # pix_theta_k = pix_theta_k % (2 * np.pi)

    return pix_mag, pix_theta_k


def kaleidoscope_pattern(
    img_input,
    pix_mag,
    pix_theta_k,
    center_in=None,
    invert_sin=False,
):

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
