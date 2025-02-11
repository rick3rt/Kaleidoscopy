import cv2

import dearpygui.dearpygui as dpg

from kaleidoscope.controller import KSettings, Kaleidoscope
from kaleidoscope.components.settings import SettingsItem, SettingsEditor
from kaleidoscope.components.texture import Texture
from kaleidoscope.utils.ratelimiter import ratelimiter

INIT_FILE = "custom_layout.ini"


class KaleidoscopeGui:
    def __init__(self, kaleidoscope):

        self.kaleidoscope = kaleidoscope
        self.textures = {}

        dpg.create_context()
        dpg.configure_app(docking=True, docking_space=True, load_init_file=INIT_FILE)
        dpg.create_viewport(title="KSpace", width=1200, height=900)
        dpg.setup_dearpygui()

        self.create_settings()
        self.create_image_view()
        self.create_menu()

    @ratelimiter(2)
    def update_image(self):
        print("UPDATING IMAG")
        print(self.textures)
        settings_new = self.settings_editor.get_values()
        if not settings_new:
            return

        self.kaleidoscope.update_settings(settings_new)
        self.kaleidoscope.run()

        # update texture
        self.textures["output"].update(self.kaleidoscope.out())
        # print(self.textures)

        # update view
        # self.create_image_view()

        # from kaleidoscope.utils import show_img
        # show_img(self.kaleidoscope.out())

    def create_settings(self):

        settings = [
            SettingsItem("num_repeats", 2, "slider_double", min=0.5, max=10),
            SettingsItem("pad_size", 100, "slider_int", min=10, max=500),
        ]
        self.settings_editor = SettingsEditor(
            settings, "Kaleidoscope Settings", callback=self.update_image
        )

    def create_image_view(self):

        with dpg.window(label="Input Image"):

            self.textures["input"] = Texture(self.kaleidoscope.input(), "input")

            with dpg.plot(label="Image Plot", height=500, width=-1, equal_aspects=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="x axis", auto_fit=True)
                with dpg.plot_axis(dpg.mvYAxis, label="y axis", auto_fit=True):
                    self.textures["input"].show_series()

            # self.textures["input"].show()

        with dpg.window(label="Output Image"):

            self.textures["output"] = Texture(self.kaleidoscope.out(), "output")

            with dpg.plot(label="Image Plot", height=500, width=-1, equal_aspects=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="x axis", auto_fit=True)
                with dpg.plot_axis(dpg.mvYAxis, label="y axis", auto_fit=True):
                    self.textures["output"].show_series()

            # self.textures["output"].show()

    def create_menu(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Layout"):
                dpg.add_menu_item(
                    label="Save",
                    callback=lambda: dpg.save_init_file(INIT_FILE),
                )

    def loop(self):
        # dpg.start_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        self.exit()

    def exit(self):
        # dpg.save_init_file(INIT_FILE)
        dpg.destroy_context()


if __name__ == "__main__":

    k_conf1 = KSettings(
        num_repeats=2,
        angle_offset_in=0,
        angle_offset_out=0,
        center_in=(100, 200),
        pad_size=(500, 500),
    )
    img = cv2.imread("image/image1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kal = Kaleidoscope(k_conf1, img)

    k = KaleidoscopeGui(kal)
    k.loop()
    # k.exit()
    print("done")
