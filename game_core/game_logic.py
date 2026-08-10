import sys
import math
import random
import pygame
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, Dict, Any
from game_core.game_objects import GameObject, Button, TextObject
from game_core.game_render import SCREEN, GAME_BG, PLAYER_THRUSTING_IMAGE, EXPLOSION_IMAGE, clock, display_image, get_font

## Rendering is off by default so rollouts are collected as fast as the machine
## allows; the menu turns it on per run. This covers test runs too, since the RL
## env always calls run_game_frame with mode "training". Manual mode always renders.
RENDER_ENABLED: bool = False

## Where the landing pad goes. Random by default; static pins it to one spot,
## which is a much easier problem for the agent to learn.
RANDOM_TARGET: bool = True
STATIC_TARGET_X: int = 590

## Both the lander and the pad spawn from this range. They used to differ, with
## the lander drawn from 200-1200 but the pad from 20-1200, which put the pad
## left of the lander in 58% of episodes by an average of 96px. The policy
## learned the bias: "left" was its most used action at 0.30 after 1M steps.
SPAWN_X_MIN: int = 20
SPAWN_X_MAX: int = 1200

def set_render_enabled(enabled: bool) -> None:
    global RENDER_ENABLED
    RENDER_ENABLED = enabled

def set_random_target(random_target: bool) -> None:
    global RANDOM_TARGET
    RANDOM_TARGET = random_target
    
K_DIST, K_SPEED = 1.0, 8.0

def phi(gs):
    d = math.hypot(gs.player.x - gs.target.x, gs.player.y - gs.target.y)
    v = math.hypot(gs.player.x_acceleration, gs.player.y_acceleration)
    return -(K_DIST * d + K_SPEED * v)

def speed(gs):
    return math.hypot(gs.player.x_acceleration, gs.player.y_acceleration)

def calculate_reward_and_done(gs, drawing: bool = True):
        reward: float = phi(gs) - gs.prev_phi

        gs.prev_phi = phi(gs)
        done: bool = False
        info: Dict[str, Any] = {}

        LANDING_acceleration_LIMIT: float = 1.0 
        if gs.player.y > 513:
            impact = speed(gs)
            if gs.player.collided_with(gs.target): 
                if impact <= LANDING_acceleration_LIMIT:
                    done = True
                    reward += 200
                    info["status"] = "landed_ok"
                else:
                    if drawing:
                        display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
                    done = True
                    reward -= 50
                    info["status"] = "too_fast"
            else:
                if drawing:
                    display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
                done = True
                reward -= 100
                info["status"] = "crashed" 

        elif gs.player.y < -50:
            if drawing:
                display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
            done = True
            reward -= 100
            info["status"] = "flown_too_high"

        elif gs.player.x < 0 or gs.player.x > 1280:
            done = True
            reward -= -100
            info["status"] = "out_of_horizontal_bounds"

        return reward, done, info


class GameState:
    def __init__(self, seed: Optional[int] = None):
        ## Own RNG rather than the global random module. With the global one,
        ## LanderEnvironment.reset(seed=...) seeded gymnasium's np_random while
        ## the game kept drawing from an unrelated stream, so seeding did
        ## nothing and runs were never reproducible.
        self.rng = random.Random(seed)
        self.player = GameObject(self.rng.randint(SPAWN_X_MIN, SPAWN_X_MAX), -30, 0, 1, "player")
        self.target = GameObject(self._target_x(), 513, 0, 0, "target")

        self.is_left_pressed: bool = False
        self.is_right_pressed: bool = False
        self.is_up_pressed: bool = False
        self.prev_phi = phi(self)

    def seed(self, seed: int) -> None:
        """Reseed the game RNG so the next reset is reproducible."""
        self.rng.seed(seed)

    def _target_x(self) -> int:
        ## Read at call time so menu choices apply to the next reset.
        return self.rng.randint(SPAWN_X_MIN, SPAWN_X_MAX) if RANDOM_TARGET else STATIC_TARGET_X

    def reset(self):
        self.player.reset()
        self.target.reset()
        self.player.x = self.rng.randint(SPAWN_X_MIN, SPAWN_X_MAX)
        self.target.x = self._target_x()
        self.player.rect.topleft = (int(self.player.x), int(self.player.y))
        self.target.rect.topleft = (int(self.target.x), int(self.target.y))
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.is_up_pressed = False
        self.prev_phi = phi(self)


    def get_observation(self) -> ndarray:
        """Build an observation from the current state without advancing the game."""
        return np.array([self.player.x - self.target.x, self.player.y - self.target.y,
                         self.player.x_acceleration, self.player.y_acceleration, self.target.x, self.target.y])

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

    LANDER_ACCELERATION: TextObject = TextObject(
    text_input=f"Acceleration: {gs.player.x_acceleration:.3f}, {gs.player.y_acceleration:.3f}",
    font=get_font(15),
    color="White",
    pos=(SCREEN.get_width() - 200, 50)
    )
    LANDER_ACCELERATION.update(SCREEN)

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

    gs.player.x_acceleration += player_x_acceleration
    gs.player.y_acceleration += player_y_acceleration
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

