from .unit import Unit
import pygame
from dataclasses import dataclass

@dataclass
class Locator(Unit):
    cover_area: float = 20
    
    # def draw(self, canvas, pix_square_size):
    #         pygame.draw.circle(
    #             canvas,
    #             (0, 0, 255, 128),
    #             (self._position + 0.5) * pix_square_size + self.cover_area,
    #             pix_square_size / 3,
    #         )