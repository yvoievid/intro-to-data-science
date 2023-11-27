from .unit import Unit
import pygame
from dataclasses import dataclass

@dataclass
class Tank(Unit):
    fuel: int = 100
    speed: int = 4
    power: int = 10
    cover_area: int = 3
    health: int = 45
    damage: int = 15