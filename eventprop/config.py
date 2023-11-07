from dataclasses import dataclass


def make_dc(config, name="d_dataclass"):
    @dataclass
    class Wrapped:
        __annotations__ = {k: type(v) for k, v in config.items()}

    Wrapped.__qualname__ = Wrapped.__name__ = name
    Wrapped.config = config

    return Wrapped
