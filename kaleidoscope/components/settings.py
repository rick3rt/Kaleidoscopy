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
        options=None,
    ):
        self.name = name
        self.id = dpg.generate_uuid()
        self.value = value
        self.type = type
        self.min = min
        self.max = max
        self.options = options

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
        elif self.type == "checkbox":
            dpg.add_checkbox(label=self.name, default_value=self.value, tag=self.id)

        elif self.type == "combo":
            dpg.add_combo(
                label=self.name,
                items=self.options,
                default_value=self.value,
                tag=self.id,
            )
        else:
            raise ValueError(
                f"Cannot make SettingsItem with type: {self.type}. Not implemented yet"
            )

    def set_callback(self, callback):
        dpg.set_item_callback(self.id, callback)

    def get_value(self):
        return dpg.get_value(self.id)

    def randomize(self):
        if "slider" in self.type:
            self.value = self.min + (self.max - self.min) * random.random()
        elif self.type == "checkbox":
            self.value = random.choice([True, False])

        dpg.set_value(self.id, self.value)
        return self.value


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
            self.data[s.name] = s.randomize()
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
        SettingsItem("checker", False, "checkbox"),
    ]
    label = "settings"

    def test_callback():
        print(se.get_values())

    se = SettingsEditor(settings, label, callback=test_callback)

    dpg.start_dearpygui()

    dpg.destroy_context()
