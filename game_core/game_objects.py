import pygame
import os



## This is from https://github.com/phildecroos/lunar_lander/blob/master/lunar_lander.py
class GameObject(pygame.sprite.Sprite):
    def __init__(self, x: float, y: float, x_speed: float, y_speed: float, name: str):
        self.image = pygame.image.load(f"assets/{name}.png")
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
        self.rect.topleft = (int(self.x), int(self.y))

    # Change the x and y values of the object
    def move(self):
        self.x += self.x_speed
        self.y += self.y_speed
        self.rect.topleft = (int(self.x), int(self.y))

    # Move the object's rect and display its image
    def display(self, screen):
        screen.blit(self.image, (int(self.x), int(self.y)))

    def collided_with(self, other_object):
        return self.rect.colliderect(other_object.rect)
    
    def reset(self):
        self.x = self.store_x
        self.y = self.store_y
        self.x_speed = self.store_x_speed
        self.y_speed = self.store_y_speed

## Button code from https://github.com/baraltech/Menu-System-PyGame
class Button():
	def __init__(self, image, pos, text_input, font, base_color, hovering_color):
		self.image = image
		self.x_pos = pos[0]
		self.y_pos = pos[1]
		self.font = font
		self.base_color, self.hovering_color = base_color, hovering_color
		self.text_input = text_input
		self.text = self.font.render(self.text_input, True, self.base_color)
		if self.image is None:
			self.image = self.text
		self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
		self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos))

	def update(self, screen):
		if self.image is not None:
			screen.blit(self.image, self.rect)
		screen.blit(self.text, self.text_rect)

	def checkForInput(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def changeColor(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.text_input, True, self.hovering_color)
		else:
			self.text = self.font.render(self.text_input, True, self.base_color)