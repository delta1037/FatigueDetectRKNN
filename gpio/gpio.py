import os


class GPIO:
    def __init__(self):
        self.__gpio_root = "/sys/class/gpio/"
        # GPIO 方向
        self.GPIO_HIGH = "1"
        self.GPIO_LOW = "0"

    def __export(self, pin):
        with open(self.__gpio_root + "export", "w") as f:
            f.write(str(pin))

    def __unexport(self, pin):
        with open(self.__gpio_root + "unexport", "w") as f:
            f.write(str(pin))

    def set_direction(self, pin, direction):
        if not os.path.exists(self.__gpio_root + "gpio" + str(pin)):
            self.__export(pin)
        with open(self.__gpio_root + "gpio" + str(pin) + "/direction", "w") as f:
            f.write(direction)

    def set_value(self, pin, value):
        with open(self.__gpio_root + "gpio" + str(pin) + "/value", "w") as f:
            f.write(str(value))

    def get_value(self, pin):
        with open(self.__gpio_root + "gpio" + str(pin) + "/value", "r") as f:
            return f.read()
