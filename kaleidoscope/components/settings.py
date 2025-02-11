import dearpygui.dearpygui as dpg
import random

# class SliderCustom:
#     pass


class SettingsItem:
    def __init__(
        self,
        name,
        value,
        type="slider_double",
        min=None,
        max=None,
    ):
        self.name = name
        self.id = dpg.generate_uuid()
        self.value = value
        self.type = type
        self.min = min
        self.max = max

    def render(self):
        if self.type == "slider_double":
            dpg.add_slider_double(
                label=self.name,
                default_value=self.value,
                min_value=self.min,
                max_value=self.max,
                tag=self.id,
            )
        elif self.type == "slider_int":
            dpg.add_slider_int(
                label=self.name,
                default_value=self.value,
                min_value=self.min,
                max_value=self.max,
                tag=self.id,
            )

    def set_callback(self, callback):
        dpg.set_item_callback(self.id, callback)

    def get_value(self):
        return dpg.get_value(self.id)


class SettingsEditor:
    def __init__(
        self,
        settings: list,
        label: str,
        callback=None,
        callback_calls_per_second=10,
    ):
        self.settings = settings
        self.label = label
        self.data = {}

        if not callback:
            callback = self.get_values
        self.callback = callback
        self.callback_calls_per_second = callback_calls_per_second

        # internal
        self._last_call_time = 0

        with dpg.window(label=self.label):
            for s in self.settings:
                self.data[s.name] = s.get_value()
                s.render()
                s.set_callback(self.callback)

            dpg.add_button(label="Process", callback=self.callback)
            dpg.add_button(label="Randomize", callback=self.randomize)

    def randomize(self):
        for s in self.settings:
            s.value = s.min + (s.max - s.min) * random.random()
            dpg.set_value(s.id, s.value)
            self.data[s.name] = s.value
        # self.get_values()
        if self.callback:
            self.callback()

    def get_values(self):
        for s in self.settings:
            self.data[s.name] = s.get_value()
        return self.data


if __name__ == "__main__":

    dpg.create_context()

    dpg.create_viewport(title="Test Setting Editor", width=600, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    settings = {
        "num_repeats": 2,
        "pad_size": 100,
    }

    settings = [
        SettingsItem("num_repeats", 2, "slider_double", min=0.5, max=10),
        SettingsItem("pad_size", 100, "slider_int", min=10, max=500),
    ]
    label = "settings"

    def test_callback():
        print(se.get_values())

    se = SettingsEditor(settings, label, callback=test_callback)

    dpg.start_dearpygui()

    dpg.destroy_context()
