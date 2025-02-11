import os
import cv2

from kaleidoscope.controller import KSettings, Kaleidoscope, KaleidoscopeController
from kaleidoscope.utils import show_img_grid, show_img

image_folder = "image"
output_folder = "output"

filename = "image1.png"
# filename = "image2.png"
# filename = "image_doppler.png"
filename = "test_image.png"

# load image
img_input = cv2.imread(os.path.join(image_folder, filename))
img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)


# add padding around image
pad = 100
img_input = cv2.copyMakeBorder(
    img_input, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
)


# show_img(img_input)

# Run Kaleidoscope
kconf1 = KSettings(
    num_repeats=2,
    # angle_offset_in=0,
    # angle_offset_out=0,
)

kconf2 = KSettings(
    num_repeats=5,
)


k1 = Kaleidoscope(kconf1, img_input)

k2 = Kaleidoscope(kconf2, k1.out())

images = {
    "original": img_input,
    "blank": img_input * 0 + 255,
    "k1": k1.out(),
    "k2": k2.out(),
}
show_img_grid(images)
