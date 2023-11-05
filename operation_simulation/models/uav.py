from .unit import Unit
import pygame

class Uav(Unit):
    def __init__(self, position, name="uav",  step=1, size=1):
        super().__init__(position, name, step, size)
    
    def draw(self, canvas, pix_square_size):
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._position + 0.5) * pix_square_size,
            pix_square_size / 3,
        )