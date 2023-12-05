from dataclasses import dataclass

@dataclass
class Inference: 
    command: str = "ATTACK"
    flang: str = "CENTER"
    group: str = ""
    weather: str = "summer"
    strategy: str = "aggressive"
    train: bool = False
    simulate: bool = False
    dryrun: bool = False