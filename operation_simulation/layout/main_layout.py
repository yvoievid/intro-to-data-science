import pygame
from .button import Button

class MainLayout():
    def __init__(self, window) -> None:
        # windos settinds
        self.window_size = 1024
        self.menu_height = 50
        self.window = window
        self.train_button = None
        self.test_button = None

        
    def render(self):
        button_size = (200, 50)
        train_button_position = (1, 1)
        test_button_position = (220, 1)
        weather_condition_position = (420, 1)
        train_text = "Train"
        test_text = "Test"

        font = pygame.font.SysFont('Comic Sans MS', 30)
    
        header_menu = pygame.Surface((self.window_size, 50))
        header_menu.fill((255, 255, 255))        
        self.window.blit(header_menu, header_menu.get_rect())
        
        self.train_button = Button(self.window, train_text, train_button_position, button_size)
        self.train_button.draw()

        self.test_button = Button(self.window, test_text, test_button_position, button_size)
        self.test_button.draw()

    def is_test_button_clicked(self):
        return self.test_button.is_clicked()

    def is_train_button_clicked(self):
        is_clicked = self.train_button.is_clicked()
        return is_clicked