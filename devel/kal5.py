# %%
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

from myutil import show_img_grid, img_center_crop
from kaleidoscope_generator import kaleidoscope, create_pattern, kaleidoscope_pattern

# %%


image_folder = "image"
output_folder = "output"

filename = "image1.png"
# filename = "image2.png"
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


# %% k class


# k_conf1 = KSettings(
#     num_repeats=2,
#     angle_offset_in=0,
#     angle_offset_out=0,
#     center_in=(200, 200),
#     center_out=(250, 250),
#     pad_size=(400, 500),
# )

k_conf1 = KSettings(
    num_repeats=2,
    angle_offset_in=0,
    angle_offset_out=0,
    center_in=(100, 200),
    # center_out=None,
    pad_size=(500, 500),
)


k_conf2 = KSettings(
    num_repeats=5,
    angle_offset_in=0,
    angle_offset_out=0,
    center_in=None,
    center_out=None,
    pad_size=600,
)


k_conf3 = k_conf2.copy()
k_conf3["tile"] = True
k_conf3["pad_size"] = (350, 350)


print(k_conf2["tile"])
print(k_conf3["tile"])


k1 = Kaleidoscope(k_conf1, img_input)
k1.show()

k2 = Kaleidoscope(k_conf2, k1.out())
k2.show()

k3 = Kaleidoscope(k_conf3, k2.out())
k3.show()
print("done")


# %%


# %% Animation


class KaleidoscopeController:
    def __init__(self, kaleidoscope, control_points):
        self.kaleidoscope = kaleidoscope
        self.control_points = control_points
        self.images = []

    def run(self):

        tmpkey = list(self.control_points.keys())[0]
        for i in range(len(self.control_points[tmpkey])):
            for key, values in self.control_points.items():
                self.kaleidoscope.settings[key] = values[i]
            self.kaleidoscope.run()
            self.images.append(self.kaleidoscope.out())

    def write_video(self, filename, fps):

        from myutil import write_video

        # find smallest image
        size = self.images[0].shape
        for img in self.images:
            if img.shape[0] < size[0] or img.shape[1] < size[1]:
                size = img.shape

        for i, img in enumerate(self.images):
            tmp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            tmp = img_center_crop(tmp, size[0], size[1])
            self.images[i] = tmp
            # print(self.images[i].shape)

        write_video(filename, self.images, fps=fps)
        print("Done")


pattern_length = 200
pad_size = np.round(np.linspace(150, 400, pattern_length)).astype(int)
pad_size = [(p, p) for p in pad_size]
control_points = {
    # "num_repeats1": np.linspace(0, 10, pattern_length),
    "num_repeats": np.exp(np.linspace(1, 1.5, pattern_length) ** 1.5) - 1,
    "angle_offset_in": np.linspace(0, 2 * np.pi, pattern_length),
    "angle_offset_out": np.linspace(0, 2 * np.pi, pattern_length),
    # "pad_size": pad_size,
    "center_in": pad_size,
}


def double_pattern(control_points):
    for key, values in control_points.items():
        control_points[key] = np.concatenate([values, values[::-1]])
    return control_points


def plot_control_points(control_points):
    for key, values in control_points.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.show()


# control_points = double_pattern(control_points)

plot_control_points(control_points)


# %%
ks = KaleidoscopeController(k3, control_points)
ks.run()


# images = {}
# for i, img in enumerate(ks.images):
#     images[f"frame {i}"] = img
# show_img_grid(images)

ks.write_video("output/kaleidoscope_varying_new2.mp4", fps=30)
