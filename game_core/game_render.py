import sys
import pygame
from pygame.surface import Surface
from pygame.font import Font

pygame.init()

SCREEN_WIDTH: int = 1280
SCREEN_HEIGHT: int = 720
OBS_HEIGHT: int = 128 ## Was using this for CNN policy network but it is not working well so I gave up on it
OBS_WIDTH: int = 128
SCREEN: Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
N_CHANNELS: int = 3
BG = pygame.image.load("assets/MenuBackground.png")
GAME_BG: Surface = pygame.transform.scale(
    pygame.image.load("assets/AnotherGameBackground.jpg"), (SCREEN_WIDTH, SCREEN_HEIGHT)
)
PLAYER_THRUSTING_IMAGE: Surface = pygame.image.load("assets/player_thrusting.png")
EXPLOSION_IMAGE: Surface = pygame.image.load("assets/in_air_explosion_large.png")

pygame.display.set_caption("Lander Game")

clock: pygame.time.Clock = pygame.time.Clock()

def get_font(size: int) -> Font:
    return pygame.font.Font("assets/font.ttf", size)

def display_image(image: Surface, x: float, y: float) -> None:
    SCREEN.blit(image, (int(x), int(y)))

def quit_pygame():
    pygame.quit()
    sys.exit()

