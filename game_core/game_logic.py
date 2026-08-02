import sys
import math
import random
import pygame
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Dict, Any
from game_core.game_objects import GameObject, Button
from game_core.game_render import SCREEN, GAME_BG, PLAYER_THRUSTING_IMAGE, EXPLOSION_IMAGE, clock, display_image, get_font

## Rendering is off by default so rollouts are collected as fast as the machine
## allows; the menu turns it on per run. This covers test runs too, since the RL
## env always calls run_game_frame with mode "training". Manual mode always renders.
RENDER_ENABLED: bool = False

## Where the landing pad goes. Random by default; static pins it to one spot,
## which is a much easier problem for the agent to learn.
RANDOM_TARGET: bool = True
STATIC_TARGET_X: int = 590

def set_render_enabled(enabled: bool) -> None:
    global RENDER_ENABLED
    RENDER_ENABLED = enabled

def set_random_target(random_target: bool) -> None:
    global RANDOM_TARGET
    RANDOM_TARGET = random_target

def _target_x() -> int:
    ## Read at call time so menu choices apply to the next reset.
    return random.randint(20, 1200) if RANDOM_TARGET else STATIC_TARGET_X

## You can change this to whatever
## There are probably better ways to do this
def calculate_reward_and_done(gs, drawing: bool = True):
        # Distance based rewards
        current_x_distance: float = abs(gs.player.x - gs.target.x)
        current_y_distance: float = abs(gs.player.y - gs.target.y)
        current_hypotenuse: float = math.sqrt(current_x_distance ** 2 + current_y_distance ** 2)

        # if current_y_distance <= gs.previous_distance_y:
        #     distance_y: float = gs.previous_distance_y - current_y_distance
        #     reward: float = gs.proportionality_factor_y * distance_y
        # else:
        #     distance_y: float = current_y_distance - gs.previous_distance_y
        #     reward: float = - gs.proportionality_factor_y * distance_y * 0.5

        # if current_x_distance <= gs.previous_distance_x:
        #     distance_x: float = gs.previous_distance_x - current_x_distance
        #     reward += gs.proportionality_factor_x * distance_x
        # else:
        #     distance_x: float = current_x_distance - gs.previous_distance_x
        #     reward += -gs.proportionality_factor_x * distance_x * 0.5

        ## Symmetric on purpose: rewarding approach more than it penalises retreat
        ## lets the agent farm reward by oscillating (move away, move back, repeat)
        ## and never landing. Weighting both directions equally makes the per-step
        ## rewards telescope to k * (start_distance - end_distance), which is
        ## path independent and so cannot be farmed.
        distance_hypotenuse: float = gs.previous_hypotenuse - current_hypotenuse
        reward: float = gs.proportionality_factor_hypotenuse * distance_hypotenuse

        ## Acceration based rewards
        # current_acceleration_h: float = distance_hypotenuse/60/60


        done: bool = False
        gs.previous_distance_x = current_x_distance
        gs.previous_distance_y = current_y_distance
        gs.previous_hypotenuse = current_hypotenuse
        info: Dict[str, Any] = {}

        ## Terminal states only flag the episode as over. Resetting here would
        ## overwrite the state before run_game_frame builds the observation, so
        ## the caller was handed the next episode's spawn point as the terminal
        ## observation. Respawning belongs to reset().
        if gs.player.y > 513:
            if gs.player.collided_with(gs.target):
                done = True
                reward = 100
                info["status"] = "landed_ok"
            else:
                if drawing:
                    display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
                done = True
                reward = -100
                info["status"] = "crashed"

        elif gs.player.y < -50:
            if drawing:
                display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
            done = True
            reward = -100
            info["status"] = "flown_too_high"

        ## elif, not if: now that this reads the real position rather than a
        ## post-reset one, a frame that reaches the ground must keep its
        ## outcome instead of being relabelled out of bounds.
        elif gs.player.x < 0 or gs.player.x > 1280:
            done = True
            reward = -100
            info["status"] = "out_of_horizontal_bounds"

        return reward, done, info


class GameState:
    def __init__(self):
        self.player = GameObject(random.randint(20, 1200), -30, 0, 1, "player")
        self.target = GameObject(_target_x(), 513, 0, 0, "target")

        self.is_left_pressed: bool = False
        self.is_right_pressed: bool = False
        self.is_up_pressed: bool = False
        self.previous_distance_x: float = abs(self.player.x - self.target.x)
        self.previous_distance_y: float = abs(self.player.y - self.target.y)
        self.previous_hypotenuse: float = math.sqrt(self.previous_distance_x ** 2 + self.previous_distance_y ** 2)
        # self.final_reward = 3000.0
        self.proportionality_factor_x: float = 20
        self.proportionality_factor_y: float = 10
        self.proportionality_factor_hypotenuse: float = 20
        self.time_penalty: float = -0.001

    def reset(self):
        self.player.reset()
        self.target.reset()
        self.player.x = random.randint(200, 1200)
        self.target.x = _target_x()
        self.player.rect.topleft = (int(self.player.x), int(self.player.y))
        self.target.rect.topleft = (int(self.target.x), int(self.target.y))
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_up_pressed = False
        self.previous_distance_x = abs(self.player.x - self.target.x)
        self.previous_distance_y = abs(self.player.y - self.target.y)
        self.previous_hypotenuse = math.sqrt(self.previous_distance_x ** 2 + self.previous_distance_y ** 2)
        # self.final_reward = 2000.0

    def get_observation(self) -> ndarray:
        """Build an observation from the current state without advancing the game."""
        return np.array([self.player.x - self.target.x, self.player.y - self.target.y,
                         self.player.x_speed, self.player.y_speed, self.target.x, self.target.y])

def run_game_frame(
    gs: GameState,
    mode: str,
    action: Optional[int] = None
) -> Optional[Tuple[ndarray, float, bool, Dict[str, Any]]]:
    player_x_acceleration: float = 0.0
    player_y_acceleration: float = 0.005

    ## Drawing is skipped entirely when nothing will be shown. It is not just
    ## wasted work: with several envs in one process they all share SCREEN, so
    ## they would be scribbling over each other's frames for no reason.
    drawing: bool = mode != "training" or RENDER_ENABLED

    if drawing:
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

    if drawing:
        gs.target.display(SCREEN)
        if (gs.is_up_pressed and mode == "manual") or (action == 2 and (mode == "training" or mode == "test")):
            SCREEN.blit(PLAYER_THRUSTING_IMAGE, (int(gs.player.x), int(gs.player.y)))
        else:
            gs.player.display(SCREEN)

    reward, done, info = calculate_reward_and_done(gs, drawing)

    ## Flipping the display and capping at 60 FPS are display concerns. Applying
    ## them to training pinned rollout collection to ~62 steps/sec, which is
    ## about 4.5 hours for the 1M timesteps training_mode runs. The menu sets
    ## RENDER_ENABLED when you would rather watch than go fast.
    if drawing:
        pygame.display.update()
        clock.tick(60)

    observation: ndarray = gs.get_observation()
    return observation, reward, done, info

