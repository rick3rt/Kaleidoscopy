import cv2
import numpy as np

import matplotlib.pyplot as plt

from kaleidoscope.controller import Kaleidoscope, KSettings, KaleidoscopeController
from kaleidoscope.utils.plot import show_img_grid
from kaleidoscope.utils.export import write_video


def double_pattern(control_points):
    for key, values in control_points.items():
        control_points[key] = np.concatenate([values, values[::-1]])
    return control_points


def plot_control_points(control_points):
    for key, values in control_points.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    k_conf1 = KSettings(
        num_repeats=3,
        angle_offset_in=0,
        angle_offset_out=0,
        # center_in=(100, 200),
        # pad_size=None,
    )
    img = cv2.imread("image/image1.png")
    img = cv2.imread("image/image_doppler.png")
    # img = cv2.imread("image/sukkel.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kal = Kaleidoscope(k_conf1, img)

    # images = {
    #     "original": img,
    #     "k1": kal.out(),
    # }
    # show_img_grid(images)

    pattern_length = 400

    control_points = {
        # "num_repeats1": np.linspace(0, 10, pattern_length),
        # "num_repeats": np.exp(np.linspace(1, 1.5, pattern_length) ** 1.5) - 1,
        "angle_offset_in": np.linspace(0, 2 * np.pi, pattern_length),
        "angle_offset_out": np.linspace(0, 2 * np.pi, pattern_length),
    }

    control_points = double_pattern(control_points)
    # plot_control_points(control_points)

    kal_controller = KaleidoscopeController(kal, control_points)
    kal_controller.run()

    kal_controller.write_video("output/kaleidoscope_varying_doppler2.mp4", fps=30)
