import sys
import random
import pygame
import pygame.surfarray
from typing import Optional
from button import Button
from gameobject import GameObject

## Menu code from https://github.com/baraltech/Menu-System-PyGame
## Lander code from https://github.com/phildecroos/lunar_lander/blob/master/lunar_lander.py

pygame.init()

SCREEN = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("Menu")
clock = pygame.time.Clock()

player = GameObject(random.randint(20, 1200), -30, 0, 1, "player")
target = GameObject(random.randint(20, 1200), 513, 0, 0, "target")
player_thrusting = pygame.image.load("../assets/player_thrusting.png")
in_air_explosion_large = pygame.image.load("../assets/in_air_explosion_large.png")

BG = pygame.image.load("../assets/MenuBackground.png")
GameBG = pygame.image.load("../assets/AnotherGameBackground.jpg")

GameBG = pygame.transform.scale(GameBG, (1280, 720))

def get_font(size):
    return pygame.font.Font("../assets/font.ttf", size)

def display(image, x, y):
    SCREEN.blit(image, (int(x), int(y)))

def game_reset():
    player.reset()
    target.reset()
    player.x = random.randint(200, 1200)
    target.x = random.randint(20, 1200)

def game_loop(mode: str, action: Optional[int] = None):
    player_x_acceleration: float = 0
    player_y_acceleration: float = 0.005

    running = True
    while running:
        SCREEN.blit(GameBG, (0, 0))

        LANDER_MOUSE_POS = pygame.mouse.get_pos()
        LANDER_BACK = Button(image=None, pos=(100, 50),
                text_input="BACK", font=get_font(45), base_color="White", hovering_color="Green")

        LANDER_BACK.changeColor(LANDER_MOUSE_POS)
        LANDER_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if LANDER_BACK.checkForInput(LANDER_MOUSE_POS):
                    running = False

            if mode == "manual":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player_x_acceleration = -0.005
                    if event.key == pygame.K_RIGHT :
                        player_x_acceleration = 0.005
                    if event.key == pygame.K_UP :
                        player_y_acceleration = -0.005
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        player_x_acceleration = 0
                    if event.key == pygame.K_UP:
                        player_y_acceleration = 0.005

            elif mode == "training" or mode == "test":
                if action == 0: ## left
                    player_x_acceleration = -0.005
                    player_y_acceleration = 0.005
                elif action == 1: ## right
                    player_x_acceleration = 0.005
                    player_y_acceleration = 0.005
                elif action == 2: ## up
                    player_x_acceleration = 0
                    player_y_acceleration = -0.005
                elif action == 3: ## no action
                    player_x_acceleration = 0
                    player_y_acceleration = 0.005

        player.x_speed += player_x_acceleration
        player.y_speed += player_y_acceleration
        player.move()

        target.display()

        if player_x_acceleration == 0 and player_y_acceleration == 0.005:
            player.display()
        else:
            player.rect = player.image.get_rect()
            player.rect.move_ip(int(player.x), int(player.y))
            SCREEN.blit(player_thrusting, (int(player.x), int(player.y)))
        
        ## distance based reward
        distance: float = ((player.x - target.x) ** 2 + (player.y - target.y) ** 2) ** 0.5 
        distance_reward: float = 1 - (distance / 1500.0)

        done: bool = False
        reward: float = distance_reward 
        info: dict = {}

        if player.y > 530:
            if player.collided_with(target):
                game_reset()
                done: bool = True
                reward += 100
            else:
                display(in_air_explosion_large, player.x - 17, player.y - 18)
                game_reset()
                done: bool = True
                reward += -100

        if player.x < 0 or player.x > 1280:
            game_reset()

        pygame.display.update()

        observation = pygame.surfarray.pixels3d(SCREEN).copy()
        clock.tick(60)
        
        if mode == "training" or mode == "test":
            return observation, reward, done, info
    
    main_menu()


def manual_mode(switch: bool = False):
    if switch:
        game_loop("manual")

def training_mode(switch: bool = False):
    if switch:
        while True:
            pass

def test_mode(switch: bool = False):
    if switch:
        while True:
            pass

def main_menu():
    while True:
        SCREEN.blit(BG, (0, 0))

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(100).render("MAIN MENU", True, "#b68f40")
        MENU_RECT = MENU_TEXT.get_rect(center=(640, 100))

        MANUAL_MODE_BUTTON = Button(image=None, pos=(640, 240),
                            text_input="MANUAL MODE", font=get_font(45), base_color="#d7fcd4", hovering_color="White")
        TRAINING_MODE_BUTTON = Button(image=None, pos=(640, 340),
                            text_input="TRAINING MODE", font=get_font(45), base_color="#d7fcd4", hovering_color="White")
        TEST_MODE_BUTTON = Button(image=None, pos=(640, 440),
                            text_input="TEST MODE", font=get_font(45), base_color="#d7fcd4", hovering_color="White")
        QUIT_BUTTON = Button(image=None, pos=(640, 540),
                            text_input="QUIT", font=get_font(45), base_color="#d7fcd4", hovering_color="White")

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [MANUAL_MODE_BUTTON,  TRAINING_MODE_BUTTON, TEST_MODE_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if MANUAL_MODE_BUTTON.checkForInput(MENU_MOUSE_POS):
                    manual_mode(True)
                if TRAINING_MODE_BUTTON.checkForInput(MENU_MOUSE_POS):
                    training_mode(True)
                if TEST_MODE_BUTTON.checkForInput(MENU_MOUSE_POS):
                    test_mode(True)
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

main_menu()