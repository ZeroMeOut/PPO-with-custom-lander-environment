import pygame

## This is from https://github.com/phildecroos/lunar_lander/blob/master/lunar_lander.py
class GameObject(pygame.sprite.Sprite):
    def __init__(self, x: float, y: float, x_acceleration: float, y_acceleration: float, name: str):
        self.image = pygame.image.load(f"assets/{name}.png")
        self.store_x = x
        self.store_y = y
        self.store_x_acceleration = x_acceleration
        self.store_y_acceleration = y_acceleration
        self.x = x
        self.y = y
        self.x_acceleration = x_acceleration
        self.y_acceleration = y_acceleration
        self.name = name
        self.rect = self.image.get_rect()
        self.rect.topleft = (int(self.x), int(self.y))

    # Change the x and y values of the object
    def move(self):
        self.x += self.x_acceleration
        self.y += self.y_acceleration
        self.rect.topleft = (int(self.x), int(self.y))

    # Move the object's rect and display its image
    def display(self, screen):
        screen.blit(self.image, (int(self.x), int(self.y)))

    def collided_with(self, other_object):
        return self.rect.colliderect(other_object.rect)
    
    def reset(self):
        self.x = self.store_x
        self.y = self.store_y
        self.x_acceleration = self.store_x_acceleration
        self.y_acceleration = self.store_y_acceleration

## Made from https://github.com/baraltech/Menu-System-PyGame
class TextObject:
    def __init__(self, text_input, font, color, pos):
        self.text_input = text_input
        self.font = font
        self.color = color
        self.x_pos, self.y_pos = pos
        self.text_surface = font.render(text_input, True, color)
        self.rect = self.text_surface.get_rect(center=pos)

    def render(self, color):
        self.text_surface = self.font.render(self.text_input, True, color)

    def update(self, screen):
        screen.blit(self.text_surface, self.rect)


class Button(TextObject):
    def __init__(self, image, pos, text_input, font, base_color, hovering_color):
        super().__init__(text_input, font, base_color, pos)
        self.base_color, self.hovering_color = base_color, hovering_color
        self.image = image if image is not None else self.text_surface
        self.rect = self.image.get_rect(center=pos)
        self.text_rect = self.text_surface.get_rect(center=pos)

    def update(self, screen):
        if self.image is not None:
            screen.blit(self.image, self.rect)
        screen.blit(self.text_surface, self.text_rect)

    def checkForInput(self, position):
        return self.rect.collidepoint(position)

    def changeColor(self, position):
        self.render(self.hovering_color if self.rect.collidepoint(position) else self.base_color)