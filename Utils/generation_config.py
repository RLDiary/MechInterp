from dataclasses import dataclass

@dataclass
class GenerationConfig:
    temperature = 0.5

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)