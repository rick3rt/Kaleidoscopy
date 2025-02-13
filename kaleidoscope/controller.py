import cv2
import numpy as np

from kaleidoscope.generator import kaleidoscope, kaleidoscope_pattern, create_pattern
from kaleidoscope.utils.other import img_center_crop
from kaleidoscope.utils.plot import show_img_grid
from kaleidoscope.utils.export import write_video
from kaleidoscope.utils.pattern import pattern_hex
from kaleidoscope.image import Image


class KSettings(dict):
    def __init__(
        self,
        num_repeats,
        angle_offset_in=0,
        angle_offset_out=0,
        center_in=None,
        center_out=None,
        crop_size=None,
        invert_sine=False,
        angle_mapper=None,
        mag_mapper=None,
        tiler=None,
    ):
        self["num_repeats"] = num_repeats
        self["angle_offset_in"] = angle_offset_in
        self["angle_offset_out"] = angle_offset_out
        self["center_in"] = center_in
        self["center_out"] = center_out
        self["crop_size"] = crop_size
        self["invert_sine"] = invert_sine
        self["angle_mapper"] = angle_mapper
        self["mag_mapper"] = mag_mapper
        self["tiler"] = tiler

        self.validate()

    def validate(self):

        for key, value in self.items():
            self[key] = self.validate_setting(key, value)

    def update(self, settings):
        for key, value in settings.items():
            if key not in self:
                raise ValueError(f"Setting {key} not found in settings")
            self[key] = self.validate_setting(key, value)

    @staticmethod
    def validate_setting(key, value):
        # handle some exceptions
        if key == "crop_size" and isinstance(value, int):
            value = (value, value)

        if key == "tiler":
            tiler = value
            # check if grid_size in tiler
            if "grid_size" not in tiler:
                tiler["grid_size"] = (1, 1)
            if "crop_size" in tiler and isinstance(tiler["crop_size"], int):
                tiler["crop_size"] = (tiler["crop_size"], tiler["crop_size"])
            gs = tiler["grid_size"]
            if isinstance(gs, int):
                tiler["grid_size"] = (gs, gs)

            if "type" not in tiler:
                tiler["type"] = "rect"
            if "hex_radius" not in tiler:
                tiler["hex_radius"] = 100
            if "hex_angle" not in tiler:
                tiler["hex_angle"] = 0

            value = tiler
        return value


class Kaleidoscope:
    def __init__(self, settings: KSettings, img: np.ndarray):
        self.settings = settings
        self.image_in = img.copy()
        self.image_out = None

        self.run()

    def run(self):
        # # print("K SETTINGS: ", self.settings)
        # if self.settings["tile"]:
        #     pix_mag, pix_theta_k = create_pattern(
        #         self.image_in,
        #         self.settings["num_repeats"],
        #         self.settings["angle_offset_in"],
        #         self.settings["angle_offset_out"],
        #         self.settings["center_in"],
        #         self.settings["center_out"],
        #     )

        #     # crop output image
        #     if self.settings["pad_size"]:
        #         pix_mag = img_center_crop(pix_mag, *self.settings["pad_size"])
        #         pix_theta_k = img_center_crop(pix_theta_k, *self.settings["pad_size"])
        #         tmp = img_center_crop(self.image_in, *self.settings["pad_size"])
        #     else:
        #         tmp = self.image_in

        #     def stacker(input, size=(1, 3), axis=0):
        #         input = np.tile(input, size)
        #         input = np.concatenate([np.flip(input), input], axis=axis)
        #         input = np.concatenate([np.flip(input), input], axis=axis)
        #         return input

        #     grid_width = 4
        #     pix_mag = np.tile(pix_mag, (4, grid_width))
        #     pix_theta_k = stacker(pix_theta_k, size=(1, grid_width), axis=0)
        #     tmp = np.tile(tmp, (4, grid_width, 1))

        #     self.image_out = kaleidoscope_pattern(
        #         tmp,
        #         pix_mag,
        #         pix_theta_k,
        #         self.settings["center_in"],
        #         self.settings["invert_sine"],
        #     )

        # else:
        self.image_out = kaleidoscope(
            self.image_in,
            self.settings["num_repeats"],
            self.settings["angle_offset_in"],
            self.settings["angle_offset_out"],
            self.settings["center_in"],
            self.settings["center_out"],
        )
        if self.settings["pad_size"]:
            self.image_out = img_center_crop(self.image_out, *self.settings["pad_size"])

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

    def update_input_image(self, img):
        self.image_in = img.copy()
        self.run()


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


class KaleidoscopeMapper:
    def __init__(
        self,
        settings,
        image,
    ):
        self.image_in = image
        self.image_internal = self.image_in
        self.image_out = self.image_in.copy()
        self.pix_mag = None
        self.pix_theta_k = None

        self.settings: KSettings = settings
        self._has_run = False

    def run(self, settings=None):
        if settings:
            self.settings.update(settings)

        if self.settings["crop_size"]:
            self.image_internal = Image.crop(self.image_in, self.settings["crop_size"])

        self.create_map()
        self.apply_map()
        self.apply_tiler()
        self._has_run = True
        return self.image_out

    def apply_map(self):
        self.image_out = kaleidoscope_pattern(
            self.image_internal,
            self.pix_mag,
            self.pix_theta_k,
            self.settings["center_in"],
            self.settings["invert_sine"],
        )

    def create_map(self):

        pix_mag, pix_theta_k = create_pattern(
            self.image_internal,
            self.settings["num_repeats"],
            self.settings["angle_offset_in"],
            self.settings["angle_offset_out"],
            self.settings["center_in"],
            self.settings["center_out"],
        )

        if self.settings["mag_mapper"]:
            pix_mag = self.settings["mag_mapper"](pix_mag)

        if self.settings["angle_mapper"]:
            pix_theta_k = self.settings["angle_mapper"](pix_theta_k)

        # if self.settings["grid_size"]:
        #     # only if not none
        #     pix_mag = self.grid(pix_mag)
        #     pix_theta_k = self.grid(pix_theta_k)
        #     self.image_internal = self.grid(self.image_in)

        self.pix_mag = pix_mag
        self.pix_theta_k = pix_theta_k

    # def grid(self, input):
    #     size = self.settings["grid_size"]
    #     if input.ndim == 3:
    #         size = (size[0], size[1], 1)
    #     input = np.concatenate([np.flip(input, 1), input], axis=1)
    #     input = np.concatenate([input, np.flip(input, 0)], axis=0)
    #     input = np.tile(input, size)
    #     return input

    def mapping(self):
        if not self._has_run:
            raise ValueError("You have to run the mapper first using .run()")
        return self.pix_mag, self.pix_theta_k

    def apply_tiler(self):
        if "tiler" not in self.settings:
            return
        t = PatternTiler(self.settings["tiler"], self.image_out)
        self.image_out = t.run()

    def set_input(self, image):
        self.image_in = image

    def get_input(self):
        return self.image_in

    def get_output(self):
        return self.image_out


class PatternTiler:
    def __init__(self, settings, image):
        self.settings = settings
        self.image = image
        self.image_out = None

        self._has_run = False

    def run(self):

        if "crop_size" in self.settings:
            print("Cropping image")
            self.image = Image.crop(self.image, self.settings["crop_size"])

        if self.settings["type"] == "none":
            self.image_out = self.image
        elif self.settings["type"] == "rect":
            self.image_out = self.tile_rect()
        elif self.settings["type"] == "hex":
            self.image_out = self.tile_hex()

        self._has_run = True
        return self.image_out

    def tile_rect(self):
        image = self.image
        grid_size = self.settings["grid_size"]
        if image.ndim == 3:
            grid_size = (grid_size[0], grid_size[1], 1)
        image = np.concatenate([np.flip(image, 1), image], axis=1)
        image = np.concatenate([image, np.flip(image, 0)], axis=0)
        image = np.tile(image, grid_size)
        return image

    def tile_hex(self):
        hex_radius = self.settings["hex_radius"]
        hex_angle = self.settings["hex_angle"]
        grid_size = self.settings["grid_size"]
        output_size = np.array(grid_size) * hex_radius * 3
        output_image = pattern_hex(
            self.image,
            hex_radius,
            hex_angle,
            output_size,
        )
        return output_image

    def get_image(self):
        if not self._has_run:
            raise ValueError("You have to run the tiler first using .run()")
        return self.image_out


def test_tiler():
    from kaleidoscope.utils.plot import show_img_grid

    img = Image.load("image/image_doppler.png")
    pt = PatternTiler(
        {
            "type": "rect",
            "grid_size": (1, 1),
        },
        img,
    )
    pt.run()

    image_out = pt.get_image()
    # show_img_grid({"input": img, "output": image_out})

    pt_hex = PatternTiler(
        {
            "type": "hex",
            "hex_radius": 250,
            "hex_angle": 0,
            "grid_size": (3, 3),
        },
        img,
    )
    pt_hex.run()

    image_out_hex = pt_hex.get_image()

    show_img_grid({"input": img, "output": image_out, "output_hex": image_out_hex})


def test_kaleidoscope():
    from kaleidoscope.utils.plot import show_img_grid

    image = Image.load("image/image_doppler.png")
    settings = KSettings(
        num_repeats=4,
        angle_offset_in=0,
        angle_offset_out=0,
        center_in=None,
        center_out=None,
        crop_size=900,
        invert_sine=False,
        tiler={"type": "rect", "grid_size": (2, 2)},
    )

    k = KaleidoscopeMapper(settings, image)
    image_out = k.run()

    show_img_grid({"input": image, "output": image_out})


def test_kaleidoscope_tiler():
    from kaleidoscope.utils.plot import show_img_grid

    image = Image.load("image/image_doppler.png")
    settings = KSettings(
        num_repeats=4,
        angle_offset_in=0.1,
        angle_offset_out=0,
        center_in=None,
        center_out=None,
        # pad_size=400,
        invert_sine=False,
    )

    k = KaleidoscopeMapper(settings, image)
    image_out = k.run()

    pattern_settings_rect = {
        "type": "rect",
        "grid_size": (2, 2),
        "crop_size": (400, 400),
    }

    pattern_settings_hex = {
        "type": "hex",
        "hex_radius": 220,
        "hex_angle": 0.2,
        "grid_size": (2, 2),
    }

    t = PatternTiler(
        pattern_settings_rect,
        image_out,
    )
    image_out_tile = t.run()

    show_img_grid(
        {
            "input": image,
            "output": image_out,
            "output_tile": image_out_tile,
        }
    )


if __name__ == "__main__":

    # test_tiler()
    test_kaleidoscope()
    # test_kaleidoscope_tiler()
