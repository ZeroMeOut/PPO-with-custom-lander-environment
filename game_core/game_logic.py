import sys
import math
import random
import pygame
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Dict, Any
from game_core.game_objects import GameObject, Button
from game_core.game_render import SCREEN, GAME_BG, PLAYER_THRUSTING_IMAGE, EXPLOSION_IMAGE, clock, display_image, get_font

class GameState:
    def __init__(self):
        self.player = GameObject(random.randint(20, 1200), -30, 0, 1, "player")
        # self.target = GameObject(random.randint(20, 1200), 513, 0, 0, "target") ## This is the original line
        self.target = GameObject(590, 513, 0, 0, "target")

        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_up_pressed = False
        self.previous_distance_x = abs(self.player.x - self.target.x)
        self.previous_distance_y = abs(self.player.y - self.target.y)
        self.previous_hypotenuse = math.sqrt(self.previous_distance_x ** 2 + self.previous_distance_y ** 2)
        self.final_reward = 3000.0
        self.proportionality_factor_x = 20
        self.proportionality_factor_y = 10
        self.proportionality_factor_hypotenuse = 20
        self.time_penalty = -0.001

    def reset(self):
        self.player.reset()
        self.target.reset()
        self.player.x = random.randint(200, 1200)
        # self.target.x = random.randint(20, 1200)
        self.target.x = 590
        self.player.rect.topleft = (int(self.player.x), int(self.player.y))
        self.target.rect.topleft = (int(self.target.x), int(self.target.y))
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_up_pressed = False
        self.previous_distance_x = abs(self.player.x - self.target.x)
        self.previous_distance_y = abs(self.player.y - self.target.y)
        self.previous_hypotenuse = math.sqrt(self.previous_distance_x ** 2 + self.previous_distance_y ** 2)
        self.final_reward = 2000.0

game_state = GameState()

def game_reset() -> None:
    game_state.reset()

def run_game_frame(
    mode: str, 
    action: Optional[int] = None
) -> Optional[Tuple[ndarray, float, bool, Dict[str, Any]]]:
    gs = game_state
    player_x_acceleration: float = 0.0
    player_y_acceleration: float = 0.005

    SCREEN.blit(GAME_BG, (0, 0))

    if mode == "manual":
        LANDER_MOUSE_POS: Tuple[int, int] = pygame.mouse.get_pos()
        LANDER_BACK: Button = Button(
            image=None, 
            pos=(100, 50),
            text_input="BACK", 
            font=get_font(45), 
            base_color="White", 
            hovering_color="Green"
        )
        LANDER_BACK.changeColor(LANDER_MOUSE_POS)
        LANDER_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if LANDER_BACK.checkForInput(LANDER_MOUSE_POS):
                    return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player_x_acceleration = -0.005
                    gs.is_left_pressed = True
                if event.key == pygame.K_RIGHT:
                    player_x_acceleration = 0.005
                    gs.is_right_pressed = True
                if event.key == pygame.K_UP:
                    player_y_acceleration = -0.005
                    gs.is_up_pressed = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    gs.is_left_pressed = False
                if event.key == pygame.K_RIGHT:
                    gs.is_right_pressed = False
                if event.key == pygame.K_UP:
                    gs.is_up_pressed = False

    if mode == "manual":
        if gs.is_left_pressed:
            player_x_acceleration = -0.005
        elif gs.is_right_pressed:
            player_x_acceleration = 0.005
        if gs.is_up_pressed:
            player_y_acceleration = -0.005
        else:
            player_y_acceleration = 0.005

    elif mode in ["training", "test"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()    

        if action == 0:  # left
            player_x_acceleration = -0.005
            player_y_acceleration = 0.005
        elif action == 1:  # right
            player_x_acceleration = 0.005
            player_y_acceleration = 0.005
        elif action == 2:  # up
            player_x_acceleration = 0.0
            player_y_acceleration = -0.005
        elif action == 3:  # up left
            player_x_acceleration = -0.005
            player_y_acceleration = -0.005
        elif action == 4:  # up right
            player_x_acceleration = 0.005
            player_y_acceleration = -0.005
        elif action == 5:  # no action
            player_x_acceleration = 0.0
            player_y_acceleration = 0.005

    gs.player.x_speed += player_x_acceleration
    gs.player.y_speed += player_y_acceleration
    gs.player.move()
    gs.player.rect.topleft = (int(gs.player.x), int(gs.player.y))
    gs.target.display(SCREEN)

    if (gs.is_up_pressed and mode == "manual") or (action == 2 and (mode == "training" or mode == "test")): 
        SCREEN.blit(PLAYER_THRUSTING_IMAGE, (int(gs.player.x), int(gs.player.y)))
    else:
        gs.player.display(SCREEN)

    ## Distance based rewards
    current_x_distance: float = abs(gs.player.x - gs.target.x)
    current_y_distance: float = abs(gs.player.y - gs.target.y)
    current_hypotenuse: float = math.sqrt(current_x_distance ** 2 + current_y_distance ** 2)

    if current_y_distance <= gs.previous_distance_y:
        reward: float = gs.proportionality_factor_y * (gs.previous_distance_y - current_y_distance)
    else:
        reward: float = - gs.proportionality_factor_y * (current_y_distance - gs.previous_distance_y) * 0.5

    if current_x_distance <= gs.previous_distance_x:
        reward += gs.proportionality_factor_x * (gs.previous_distance_x - current_x_distance)
    else:
        reward += -gs.proportionality_factor_x * (current_x_distance - gs.previous_distance_x) * 0.5

    if current_hypotenuse <= gs.previous_hypotenuse:
        reward += gs.proportionality_factor_hypotenuse * (gs.previous_hypotenuse - current_hypotenuse)


    ## Time penalty
    penalty: float = gs.final_reward * gs.time_penalty 
    gs.final_reward += penalty

    done: bool = False
    gs.previous_distance_x = current_x_distance
    gs.previous_distance_y = current_y_distance
    gs.previous_hypotenuse = current_hypotenuse
    info: Dict[str, Any] = {}

    if gs.player.y > 513:
        if gs.player.collided_with(gs.target):
            game_reset()
            done = True
            reward = 100
            info["status"] = "landed_ok"
        else:
            display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18) 
            game_reset()
            done = True
            reward = -100 
            info["status"] = "crashed"

    elif gs.player.y < -50:
        display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18) 
        game_reset()
        done = True
        reward = -100 
        info["status"] = "flown_too_high"

    if gs.player.x < 0 or gs.player.x > 1280:
        game_reset()
        done = True 
        reward = -100 
        info["status"] = "out_of_horizontal_bounds"

    pygame.display.update() 

    observation: ndarray = np.array([gs.player.x, gs.player.y, gs.player.x_speed, gs.player.y_speed, gs.target.x, gs.target.y])
    clock.tick(60)
    return observation, reward, done, info

