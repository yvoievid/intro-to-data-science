from .unit import Unit
from dataclasses import dataclass

@dataclass
class Soldier(Unit):
    ammo: int = 30
    cover_area: int = 1