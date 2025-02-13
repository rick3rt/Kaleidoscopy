import cv2
import time
import numpy as np
import dearpygui.dearpygui as dpg

from kaleidoscope.controller import KSettings, Kaleidoscope, KaleidoscopeMapper
from kaleidoscope.components.settings import SettingsItem, SettingsEditor
from kaleidoscope.components.dialog import (
    open_save_dialog_native,
    open_file_dialog_native,
)
from kaleidoscope.components.texture import Texture
from kaleidoscope.utils.ratelimiter import ratelimiter

INIT_FILE = "custom_layout.ini"


class KaleidoscopeGui:
    def __init__(self, kaleidoscope):

        self.kaleidoscope: KaleidoscopeMapper = kaleidoscope
        self.textures: dict[Texture] = {}

        dpg.create_context()
        dpg.configure_app(docking=True, docking_space=True, load_init_file=INIT_FILE)
        dpg.create_viewport(title="KSpace", width=1200, height=900)
        dpg.setup_dearpygui()

        self.create_settings()
        self.create_image_view()
        self.create_menu()

    @ratelimiter(10)
    def update_image(self):
        # print("update image start")
        start_time = time.time()
        kal_settings = self.settings_editor.get_values()
        tiler_settings = self.tile_editor.get_values()
        if not kal_settings and not tiler_settings:
            return

        if tiler_settings:
            kal_settings["tiler"] = tiler_settings

        image_out = self.kaleidoscope.run(kal_settings)
        self.textures["output"].update(image_out)  # update texture

        print(f"update image took {time.time() - start_time:.2f}s")

    def create_settings(self):

        settings = [
            SettingsItem("num_repeats", 2, "slider_int", min=1, max=15),
            # SettingsItem("pad_size", 500, "slider_int", min=400, max=600),
            SettingsItem("angle_offset_in", 0, "slider_double", min=0, max=np.pi * 2),
            SettingsItem("angle_offset_out", 0, "slider_double", min=0, max=np.pi * 2),
            SettingsItem("crop_size", 500, "slider_int", min=100, max=500),
            # SettingsItem("tiler", False, "checkbox"),
        ]
        self.settings_editor = SettingsEditor(
            settings, "Kaleidoscope Settings", callback=self.update_image
        )

        self.tile_editor = SettingsEditor(
            [
                SettingsItem("type", "rect", "combo", options=["none", "rect", "hex"]),
                SettingsItem("grid_size", 1, "slider_int", min=1, max=10),
                SettingsItem("crop_size", 200, "slider_int", min=100, max=500),
            ],
            "Tiler Settings",
            callback=self.update_image,
        )

    def create_image_view(self):
        with dpg.window(label="Input Image"):

            self.textures["input"] = Texture(self.kaleidoscope.get_input(), "input")
            with dpg.plot(label="Image Plot", height=-1, width=-1, equal_aspects=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="x axis", auto_fit=True)
                with dpg.plot_axis(dpg.mvYAxis, label="y axis", auto_fit=True):
                    self.textures["input"].show_series()

        with dpg.window(label="Output Image"):

            self.textures["output"] = Texture(self.kaleidoscope.get_output(), "output")
            with dpg.plot(label="Image Plot", height=-1, width=-1, equal_aspects=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="x axis", auto_fit=True)
                with dpg.plot_axis(dpg.mvYAxis, label="y axis", auto_fit=True):
                    self.textures["output"].show_series()

    def loop(self):
        dpg.show_viewport()

        # dpg.start_dearpygui()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        self.exit()

    def exit(self):
        dpg.destroy_context()

    def create_menu(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(
                    label="Load Image",
                    callback=self.load_image,
                )
                dpg.add_menu_item(
                    label="Save Image",
                    callback=self.save_image,
                )
            with dpg.menu(label="Layout"):
                dpg.add_menu_item(
                    label="Save",
                    callback=lambda: dpg.save_init_file(INIT_FILE),
                )

    def load_image(self):
        path = open_file_dialog_native()
        if path:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.kaleidoscope.set_input(img)
            self.textures["input"].update(self.kaleidoscope.get_input())

    def save_image(self):
        path = open_save_dialog_native()
        if path:
            img = self.kaleidoscope.out()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)


if __name__ == "__main__":

    k_conf1 = KSettings(
        num_repeats=2,
        angle_offset_in=0,
        angle_offset_out=0,
        tiler={"type": "rect", "size": (3, 3)},
    )
    img = cv2.imread("image/image1.png")
    # img = cv2.imread("image/image_doppler.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kal = KaleidoscopeMapper(k_conf1, img)

    k = KaleidoscopeGui(kal)
    k.loop()
    # k.exit()
    print("done")
