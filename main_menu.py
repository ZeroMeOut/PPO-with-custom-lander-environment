import os
import sys
import time
import pygame
from game_core.game_objects import Button
from game_core.game_logic import run_game_frame, game_reset 
from game_core.game_render import SCREEN, get_font, BG, quit_pygame

from stable_baselines3 import PPO
from env.lander_env import LanderEnvironment
from stable_baselines3.common.env_checker import check_env

 
## Made with the help of Gemini
## Check dummy_test_code for the skeleton of the game
## Menu code from https://github.com/baraltech/Menu-System-PyGame
def manual_mode():
    # In manual mode, run_game_frame needs to be called in a loop
    # and handle its own events.
    game_reset()  # Reset the game state before entering manual mode
    while True:
        result = run_game_frame("manual")
        if result is not None:
            print(f"Manual mode result: {result[1]}") ## Degugging reward value
        if result is None: # game_loop returns None if BACK is pressed
            break # Exit manual mode loop

## You can watch sentdex's video on this, pretty cool
## https://www.youtube.com/watch?v=uKnjGn8fF70&t=540s
def training_mode():
    if not pygame.get_init():
        pygame.init()

    current_time_str = str(int(time.time()))
    models_dir = f"models/{current_time_str}/"
    logdir = f"logs/{current_time_str}/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    env = LanderEnvironment()
    check_env(env, warn=True, skip_render_check=True)
    env.render_mode = 'human'

    ## I got the congigs from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    ## Thanks to this blog for linking the configs: https://antoinebrl.github.io/blog/rl-mars-lander/#reward-shaping
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device='cpu', n_steps=1024, 
                gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01)
    TIMESTEPS = 1000000
    iters = 0
    while iters < 10:  # Run for 10 iterations
        iters += 1
        print(f"Training iteration: {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_Lander")
        model.save(f"{models_dir}/{iters}")
        print(f"Model saved to {models_dir}/{TIMESTEPS*iters}")

        # Escape mechanism to return to menu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    env.close()
                    if not pygame.get_init():
                        pygame.init()
                    main_menu() # Go back to main menu
                    return # Exit training_mode function

    print("Training finished!")
    if not pygame.get_init():
        pygame.init()
    main_menu()

## Test mode till a work in progress
def test_mode():
    if not pygame.get_init():
        pygame.init()
    while True:
        LANDER_MOUSE_POS = pygame.mouse.get_pos()
        SCREEN.fill("black")
        LANDER_TEXT = get_font(45).render("This is the TEST game screen.", True, "White")
        LANDER_RECT = LANDER_TEXT.get_rect(center=(640, 260))
        SCREEN.blit(LANDER_TEXT, LANDER_RECT)

        LANDER_BACK = Button(image=None, pos=(640, 460),
                            text_input="BACK", font=get_font(75), base_color="White", hovering_color="Green")
        LANDER_BACK.changeColor(LANDER_MOUSE_POS)
        LANDER_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if LANDER_BACK.checkForInput(LANDER_MOUSE_POS):
                    main_menu()
        pygame.display.update()

def main_menu():
    if not pygame.get_init():
        pygame.init()
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

        for button in [MANUAL_MODE_BUTTON, TRAINING_MODE_BUTTON, TEST_MODE_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if MANUAL_MODE_BUTTON.checkForInput(MENU_MOUSE_POS):
                    manual_mode() # Call without 'True' switch
                if TRAINING_MODE_BUTTON.checkForInput(MENU_MOUSE_POS):
                    training_mode() # Call without 'True' switch
                if TEST_MODE_BUTTON.checkForInput(MENU_MOUSE_POS):
                    test_mode() # Call without 'True' switch
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()

if __name__ == "__main__":
    main_menu()
