import pygame
import os

current_path = os.path.dirname(__file__)
SCREEN = pygame.display.set_mode((1280, 720))

## This is from https://github.com/phildecroos/lunar_lander/blob/master/lunar_lander.py
class GameObject(pygame.sprite.Sprite):
    def __init__(self, x: float, y: float, x_speed: float, y_speed: float, name: str):
        self.image = pygame.image.load(
             "../assets/" + name + ".png"
        )
        self.store_x = x
        self.store_y = y
        self.store_x_speed = x_speed
        self.store_y_speed = y_speed
        self.x = x
        self.y = y
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.name = name
        self.rect = self.image.get_rect()

    # Change the x and y values of the object
    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed

    # Move the object's rect and display its image
    def display(self):
        self.rect = self.image.get_rect()
        self.rect.move_ip(int(self.x), int(self.y))
        SCREEN.blit(self.image, (int(self.x), int(self.y)))

    def collided_with(self, other_object):
        return self.rect.colliderect(other_object.rect)
    
    def reset(self):
        self.x = self.store_x
        self.y = self.store_y
        self.x_speed = self.store_x_speed
        self.y_speed = self.store_y_speed

