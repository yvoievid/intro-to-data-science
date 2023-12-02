import pygame

class Button():
    def __init__(self, screen, text, position, size):
         self.rect = pygame.Rect(position[0], position[1], 100, 100)
         self.font = pygame.font.SysFont('Comic Sans MS', 30)
         self.clicked = False  
         self.screen = screen
         self.button_size = size
         self.updateText(text)


    def updateText(self, text):
         self.text = text   
         self.render = self.font.render(self.text, True, (0, 122, 0))
         self.box = pygame.Surface((self.button_size)) 
         self.rect = self.render.get_rect(topleft = self.rect.topleft)
         self.box.fill((255, 10, 255))
        
    def draw(self):       
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            self.box.set_alpha(100) 
            self.box.fill((255, 10, 255))
            
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
        
            if pygame.mouse.get_pressed()[0] == 0:
                self.clicked = False
        else:
            self.box.set_alpha(1) 

        self.box.fill((150, 150, 150))
        self.screen.blit(self.box, (self.rect.x, self.rect.y))
        self.screen.blit(self.render, (self.rect.x, self.rect.y))   
            
        return self.clicked
    
    
    def is_clicked(self):
        pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(pos):
            self.box.set_alpha(100) 
            self.box.fill((255, 10, 255))
            
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
        
            if pygame.mouse.get_pressed()[0] == 0:
                self.clicked = False
    
        return self.clicked
    
    