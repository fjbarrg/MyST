import pandas as pd
from typing import Optional

from functions import Passive


class Main:

    @staticmethod
    def hello(name: Optional[str] = None):
        if name is None:
            name = "world"
        return f"Hello, {name}!"

    @staticmethod
    def get_passive_pre_pandemic():
        return print(Passive().get_pre_pandemic())

    @staticmethod
    def get_passive_in_pandemic():
        return print(Passive().get_in_pandemic())
