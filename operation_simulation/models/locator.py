from .unit import Unit
from dataclasses import dataclass

@dataclass
class Locator(Unit):
    cover_area: float = 20