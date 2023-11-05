from .unit import Unit
import pygame

class Locator(Unit):
    def __init__(self, position, name="locator",  step=1, size=1, area_covered=10):
        super().__init__(position, name, step, size)
        self._area_covered = area_covered
    
    def draw(self, canvas, pix_square_size):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self._position + 0.5) * pix_square_size,
                pix_square_size / 3,
            )