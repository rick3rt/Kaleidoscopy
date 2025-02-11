import cv2
import numpy as np

from kaleidoscope.generator import kaleidoscope, kaleidoscope_pattern, create_pattern
from kaleidoscope.utils.other import img_center_crop
from kaleidoscope.utils.plot import show_img_grid


class KSettings(dict):
    def __init__(
        self,
        num_repeats,
        angle_offset_in=0,
        angle_offset_out=0,
        center_in=None,
        center_out=None,
        pad_size=None,
        tile=False,
        invert_sine=False,
    ):
        self["num_repeats"] = num_repeats
        self["angle_offset_in"] = angle_offset_in
        self["angle_offset_out"] = angle_offset_out
        self["center_in"] = center_in
        self["center_out"] = center_out
        self["pad_size"] = pad_size
        self["tile"] = tile
        self["invert_sine"] = invert_sine

        if isinstance(self["pad_size"], int):
            self["pad_size"] = (self["pad_size"], self["pad_size"])


class Kaleidoscope:
    def __init__(self, settings: KSettings, img: np.ndarray):
        self.settings = settings
        self.image_in = img.copy()
        self.image_out = None

        self.run()

    def run(self):
        if self.settings["tile"]:
            pix_mag, pix_theta_k = create_pattern(
                self.image_in,
                self.settings["num_repeats"],
                self.settings["angle_offset_in"],
                self.settings["angle_offset_out"],
                self.settings["center_in"],
                self.settings["center_out"],
            )

            # crop output image
            if self.settings["pad_size"]:
                pix_mag = img_center_crop(pix_mag, *self.settings["pad_size"])
                pix_theta_k = img_center_crop(pix_theta_k, *self.settings["pad_size"])
                tmp = img_center_crop(self.image_in, *self.settings["pad_size"])

            def stacker(input, size=(1, 3), axis=0):
                input = np.tile(input, size)
                input = np.concatenate([np.flip(input), input], axis=axis)
                input = np.concatenate([np.flip(input), input], axis=axis)
                return input

            grid_width = 4
            pix_mag = np.tile(pix_mag, (4, grid_width))
            pix_theta_k = stacker(pix_theta_k, size=(1, grid_width), axis=0)
            tmp = np.tile(tmp, (4, grid_width, 1))

            self.image_out = kaleidoscope_pattern(
                tmp,
                pix_mag,
                pix_theta_k,
                self.settings["center_in"],
                self.settings["invert_sine"],
            )

        else:
            self.kaleidoscope()
            if self.settings["pad_size"]:
                self.image_out = img_center_crop(
                    self.image_out, *self.settings["pad_size"]
                )

    def kaleidoscope(self):
        self.image_out = kaleidoscope(
            self.image_in,
            self.settings["num_repeats"],
            self.settings["angle_offset_in"],
            self.settings["angle_offset_out"],
            self.settings["center_in"],
            self.settings["center_out"],
        )

    def input(self):
        return self.image_in

    def out(self):
        return self.image_out

    def show(self):
        images = {
            f"input {self.image_in.shape}": self.image_in,
            f"output {self.image_out.shape}": self.image_out,
        }
        show_img_grid(images)

    def update_settings(self, settings):
        for key, value in settings.items():
            if key not in self.settings:
                continue

            # handle some exceptions
            if key == "pad_size" and isinstance(value, int):
                value = (value, value)

            self.settings[key] = value


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

        from utils import write_video

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
