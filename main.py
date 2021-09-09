from typing import Optional
from functions import Passive, Active, Metrics


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


if __name__ == "__main__":
    print(Passive().get_pre_pandemic())
    print(Passive().get_in_pandemic())
    print(Active().get_in_pandemic())
    print(Active().get_historical_operations())
    print(Metrics().get_metrics())
    #Correrlo con python -m main
