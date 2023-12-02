from dataclasses import dataclass

@dataclass
class Inference: 
    strategy: str = "ATTACK"
    flang: str = "CENTER"
    train: bool = False
    simulate: bool = False