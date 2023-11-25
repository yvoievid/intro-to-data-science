from .unit import Unit
import pygame
from dataclasses import dataclass

@dataclass
class Locator(Unit):
    cover_area: float = 20