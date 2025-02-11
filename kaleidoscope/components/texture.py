import dearpygui.dearpygui as dpg
import numpy as np
import cv2


class Texture:
    def __init__(self, img, name="texture"):
        # get width and height
        self.name = name
        self.img = None
        self.size = None
        self.texture_id = None
        self.texture_data = []

        self._show_image = None
        self._show_series = None

        self._create_texture(img)

    def show(self):
        self._show_image = dpg.add_image(self.texture_id)

    def show_series(self):
        h, w = self.img.shape[:2]
        self._show_series = dpg.add_image_series(
            self.texture_id, [0, 0], [w, h], label=self.name
        )

    def update(self, img):
        if self.size[:2] != img.shape[:2]:
            # need to delete id and recreate texture with correct size
            # raise ValueError("Image size does not match texture size")
            dpg.delete_item(self.texture_id)
            self._create_texture(img)
        else:
            self.img = img
            self.texture_data = self._texture_data()
            dpg.set_value(self.texture_id, self.texture_data)

        if self._show_image:
            dpg.configure_item(
                self._show_image,
                texture_tag=self.texture_id,
                width=self.size[1],
                height=self.size[0],
            )
        if self._show_series:
            dpg.configure_item(
                self._show_series,
                texture_tag=self.texture_id,
                bounds_min=[0, 0],
                bounds_max=[self.size[1], self.size[0]],
            )

    def _create_texture(self, img):
        # handle recreating and deleting
        self.img = img
        self.size = img.shape
        if not self.texture_id:
            self.texture_id = dpg.generate_uuid()
        # self.texture_id = dpg.generate_uuid()
        self.texture_data = self._texture_data()
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                self.size[1],
                self.size[0],
                default_value=self.texture_data,
                tag=self.texture_id,
            )

    def _texture_data(self):
        img = self.img
        if img.shape[2] == 3:
            # img[:, :, 3] = 255  # add alph channel to rgb
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img = img / 255
        texture_data = img.flatten().tolist()
        return texture_data

        # texture_data = []
        # for i in range(0, h * w):
        #     x, y = divmod(i, w)
        #     texture_data.append(img[x, y, 0])  # R
        #     texture_data.append(img[x, y, 1])  # G
        #     texture_data.append(img[x, y, 2])  # B
        #     if self.size[2] == 4:
        #         alpha = img[x, y, 3]
        #     else:
        #         alpha = 1.0
        #     texture_data.append(alpha)  # Alpha
        # return texture_data

    def __str__(self):
        return f"Texture - size{self.size} - id: {self.texture_id}"

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    import cv2

    dpg.create_context()

    dpg.create_viewport(title="Test Setting Editor", width=600, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    img = cv2.imread("image/image1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    texture = Texture(img, "test")

    def load_img1():
        print("load img1")
        img = cv2.imread("image/image1.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f" -  {texture}")
        texture.update(img)
        print(f" +  {texture}")

        # texture.show()

    def load_img2():
        print("load img2")
        # img = cv2.imread("image/image1_size.png")
        img = cv2.imread("image/image_test.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f" -  {texture}")
        texture.update(img)
        print(f" +  {texture}")

    with dpg.window(label="image window"):

        # texture.show()

        with dpg.plot(label="Image Plot", height=500, width=500, equal_aspects=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="x axis", auto_fit=True)
            with dpg.plot_axis(dpg.mvYAxis, label="y axis", auto_fit=True):
                texture.show_series()

        dpg.add_button(label="img1", callback=load_img1)
        dpg.add_button(label="img2", callback=load_img2)

    dpg.start_dearpygui()

    dpg.destroy_context()
