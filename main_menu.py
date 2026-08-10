import os
import sys
import time
import glob
import pygame
import numpy as np
from game_core.game_objects import Button
from game_core.game_logic import run_game_frame, GameState, set_render_enabled, set_random_target
from game_core.game_render import SCREEN, get_font, BG, quit_pygame

from stable_baselines3 import PPO
from env.lander_env import LanderEnvironment
from env.callbacks import ActionFrequencyCallback, EpisodeOutcomeCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

 
## Made with the help of Gemini
## Check dummy_test_code for the skeleton of the game
## Menu code from https://github.com/baraltech/Menu-System-PyGame
def options_screen(title, toggles, start_label="START", extra_lines=()):
    """Screen of on/off options shown before a mode starts.

    toggles: list of (key, label, default) describing each option.
    Returns {key: bool} once the user starts, or None if they back out.
    """
    values = {key: default for key, _, default in toggles}

    while True:
        SCREEN.fill("black")

        TITLE_TEXT = get_font(50).render(title, True, "#b68f40")
        SCREEN.blit(TITLE_TEXT, TITLE_TEXT.get_rect(center=(640, 80)))

        for i, line in enumerate(extra_lines):
            LINE_TEXT = get_font(25).render(line, True, "Gray")
            SCREEN.blit(LINE_TEXT, LINE_TEXT.get_rect(center=(640, 140 + i * 30)))

        MOUSE_POS = pygame.mouse.get_pos()
        top = 140 + len(extra_lines) * 30 + 50

        buttons = []
        for i, (key, label, _) in enumerate(toggles):
            on = values[key]
            button = Button(image=None, pos=(640, top + i * 60),
                            text_input=f"{label}: {'ON' if on else 'OFF'}",
                            font=get_font(32),
                            base_color="Green" if on else "Red", hovering_color="White")
            button.changeColor(MOUSE_POS)
            button.update(SCREEN)
            buttons.append((key, button))

        y = top + len(toggles) * 60 + 40
        START_BUTTON = Button(image=None, pos=(640, y), text_input=start_label,
                              font=get_font(40), base_color="Green", hovering_color="White")
        BACK_BUTTON = Button(image=None, pos=(640, y + 65), text_input="BACK",
                             font=get_font(40), base_color="White", hovering_color="Green")
        for button in (START_BUTTON, BACK_BUTTON):
            button.changeColor(MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for key, button in buttons:
                    if button.checkForInput(MOUSE_POS):
                        values[key] = not values[key]
                if START_BUTTON.checkForInput(MOUSE_POS):
                    return values
                if BACK_BUTTON.checkForInput(MOUSE_POS):
                    return None

        pygame.display.update()

def manual_mode():
    options = options_screen("MANUAL MODE",
                             [("random_target", "Random target position", True)],
                             start_label="PLAY")
    if options is None:
        return  # Back to main menu
    set_random_target(options["random_target"])

    # In manual mode, run_game_frame needs to be called in a loop
    # and handle its own events.
    gs = GameState()  # Manual mode owns its own game state
    while True:
        result = run_game_frame(gs, "manual")
        if result is None: # game_loop returns None if BACK is pressed
            break # Exit manual mode loop
        _, _, done, _ = result
        if done: # crashing and landing no longer respawn from inside the reward function
            gs.reset()

## You can watch sentdex's video on this, pretty cool
## https://www.youtube.com/watch?v=uKnjGn8fF70&t=540s
def training_mode():
    if not pygame.get_init():
        pygame.init()

    options = options_screen("TRAINING OPTIONS", [
        ("render", "Render while training (much slower)", False),
        ("random_target", "Random target position", True),
    ], extra_lines=("Rendering caps training at ~60 steps/sec.",
                    "Leave it off unless you want to watch."))
    if options is None:
        return  # Back to main menu
    set_render_enabled(options["render"])
    set_random_target(options["random_target"])

    ## Name the run after how the target was placed, so the two kinds of run stay
    ## distinguishable in models/ and in tensorboard.
    target_mode = "random_target_training" if options["random_target"] else "static_target_training"
    current_time_str = str(int(time.time()))
    models_dir = f"models/{current_time_str}_{target_mode}/"
    logdir = f"logs/{current_time_str}_{target_mode}/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)
    
    check_env(LanderEnvironment(), warn=True, skip_render_check=True)

    ## Running several envs in one process is only safe now that each owns its
    ## game state. It is ~2.9x faster end to end, because the policy sees one
    ## batch of 8 observations instead of 8 separate forward passes. Rendering
    ## forces a single env, since they would all draw over the same screen.
    n_envs = 1 if options["render"] else 8
    env = make_vec_env(LanderEnvironment, n_envs=n_envs)

    ## I got the congigs from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    ## Thanks to this blog for linking the configs: https://antoinebrl.github.io/blog/rl-mars-lander/#reward-shaping
    ## n_steps is per env, so divide it to keep the rollout buffer at 2048 transitions.
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device='cpu',
                n_steps=2048 // n_envs,
                gae_lambda=0.98, gamma=0.999, n_epochs=10, ent_coef=0.05, vf_coef=0.5)
    ## Two things ep_rew_mean cannot tell you: which actions the policy favours,
    ## and how often it actually lands rather than merely getting close.
    training_callbacks = CallbackList([ActionFrequencyCallback(), EpisodeOutcomeCallback()])

    TIMESTEPS = 10000000
    iters = 0
    while iters < 10:  # Run for 10 iterations
        iters += 1
        print(f"Training iteration: {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_Lander",
                    callback=training_callbacks)
        model.save(f"{models_dir}/{iters}")
        print(f"Model saved to {models_dir}{iters}.zip after {TIMESTEPS*iters} timesteps")

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

## I got bored and told Claude to write the test model section for me
## It did a pretty good job first try so I am just gonna leave it mostly as is
def get_available_models():
    """Get all available trained models from the models directory."""
    models = []
    if os.path.exists("models"):
        model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
        for model_dir in model_dirs:
            model_path = os.path.join("models", model_dir)
            model_files = glob.glob(os.path.join(model_path, "*.zip"))
            if model_files:
                # Get the highest iteration number for this training session
                iterations = []
                for model_file in model_files:
                    try:
                        filename = os.path.basename(model_file).replace('.zip', '')
                        iteration = int(filename)
                        iterations.append(iteration)
                    except ValueError:
                        continue
                if iterations:
                    max_iteration = max(iterations)
                    best_model_path = os.path.join(model_path, f"{max_iteration}.zip")
                    ## Runs are named "<timestamp>_<random|static>_target_training".
                    ## Older runs are a bare timestamp, so fall back to that.
                    if "_random_target" in model_dir:
                        label = f"{model_dir.split('_')[0]} random target"
                    elif "_static_target" in model_dir:
                        label = f"{model_dir.split('_')[0]} static target"
                    else:
                        label = model_dir
                    models.append({
                        'name': f"{label} (Iter {max_iteration})",
                        'path': best_model_path,
                        'timestamp': model_dir
                    })
    return models

def model_selection_screen():
    """Screen to select which trained model to test."""
    models = get_available_models()
    
    if not models:
        # No models available screen
        while True:
            SCREEN.fill("black")
            
            NO_MODELS_TEXT = get_font(45).render("No trained models found!", True, "Red")
            NO_MODELS_RECT = NO_MODELS_TEXT.get_rect(center=(640, 200))
            SCREEN.blit(NO_MODELS_TEXT, NO_MODELS_RECT)
            
            INSTRUCTION_TEXT = get_font(30).render("Train a model first using TRAINING MODE", True, "White")
            INSTRUCTION_RECT = INSTRUCTION_TEXT.get_rect(center=(640, 250))
            SCREEN.blit(INSTRUCTION_TEXT, INSTRUCTION_RECT)
            
            MOUSE_POS = pygame.mouse.get_pos()
            BACK_BUTTON = Button(image=None, pos=(640, 400),
                                text_input="BACK", font=get_font(45), 
                                base_color="White", hovering_color="Green")
            BACK_BUTTON.changeColor(MOUSE_POS)
            BACK_BUTTON.update(SCREEN)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if BACK_BUTTON.checkForInput(MOUSE_POS):
                        return None
            
            pygame.display.update()
    
    # Model selection screen
    selected_model = 0
    while True:
        SCREEN.fill("black")
        
        TITLE_TEXT = get_font(60).render("SELECT MODEL TO TEST", True, "#b68f40")
        TITLE_RECT = TITLE_TEXT.get_rect(center=(640, 100))
        SCREEN.blit(TITLE_TEXT, TITLE_RECT)
        
        MOUSE_POS = pygame.mouse.get_pos()
        
        # Display model options
        buttons = []
        for i, model in enumerate(models):
            y_pos = 200 + i * 80
            color = "Yellow" if i == selected_model else "#d7fcd4"
            hover_color = "White"
            
            button = Button(image=None, pos=(640, y_pos),
                          text_input=model['name'], font=get_font(30), 
                          base_color=color, hovering_color=hover_color)
            button.changeColor(MOUSE_POS)
            button.update(SCREEN)
            buttons.append(button)
        
        # Navigation buttons
        SELECT_BUTTON = Button(image=None, pos=(640, 200 + len(models) * 80 + 50),
                              text_input="TEST THIS MODEL", font=get_font(40), 
                              base_color="Green", hovering_color="White")
        BACK_BUTTON = Button(image=None, pos=(640, 200 + len(models) * 80 + 120),
                            text_input="BACK", font=get_font(40), 
                            base_color="White", hovering_color="Green")
        
        SELECT_BUTTON.changeColor(MOUSE_POS)
        SELECT_BUTTON.update(SCREEN)
        BACK_BUTTON.changeColor(MOUSE_POS)
        BACK_BUTTON.update(SCREEN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Check model selection
                for i, button in enumerate(buttons):
                    if button.checkForInput(MOUSE_POS):
                        selected_model = i
                
                if SELECT_BUTTON.checkForInput(MOUSE_POS):
                    return models[selected_model]
                if BACK_BUTTON.checkForInput(MOUSE_POS):
                    return None
        
        pygame.display.update()

def run_test_episodes(model_path, num_episodes=10, render=True):
    """Run test episodes and collect performance metrics."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the trained model
        model = PPO.load(model_path)
        
        # Create environment
        env = LanderEnvironment()
        env.render_mode = 'human' if render else None
        
        # Statistics tracking
        episode_rewards = []
        success_count = 0
        crash_count = 0
        out_of_bounds_count = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            max_steps = 200000  # Prevent infinite episodes
            
            while not done and step_count < max_steps:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)
                ## print(type(action), action)  
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated  # a timed-out episode is over too
                episode_reward += reward
                step_count += 1
                
                # Check for pygame events to allow quitting
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return episode_rewards, success_count, crash_count, out_of_bounds_count, episode + 1
                
                if done:
                    # Count different outcomes
                    if "status" in info:
                        if info["status"] == "landed_ok":
                            success_count += 1
                        elif info["status"] == "crashed":
                            crash_count += 1
                        elif info["status"] == "out_of_horizontal_bounds":
                            out_of_bounds_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        
        # env.close()
        # print("Testing finished!") ## Commented out to avoid pygame quitting before displaying results
        return episode_rewards, success_count, crash_count, out_of_bounds_count, num_episodes
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def display_results(results):
    """Display the test results on screen."""
    if results is None:
        return
    
    episode_rewards, success_count, crash_count, out_of_bounds_count, total_episodes = results
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    success_rate = (success_count / total_episodes) * 100
    
    while True:
        SCREEN.fill("black")
        
        # Title
        TITLE_TEXT = get_font(50).render("TEST RESULTS", True, "#b68f40")
        TITLE_RECT = TITLE_TEXT.get_rect(center=(640, 50))
        SCREEN.blit(TITLE_TEXT, TITLE_RECT)
        
        # Results
        y_offset = 120
        line_spacing = 40
        
        results_text = [
            f"Episodes Completed: {total_episodes}",
            f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}",
            f"Best Episode: {max_reward:.2f}",
            f"Worst Episode: {min_reward:.2f}",
            f"",
            f"Successful Landings: {success_count} ({success_rate:.1f}%)",
            f"Crashes: {crash_count}",
            f"Out of Bounds: {out_of_bounds_count}",
        ]
        
        for i, text in enumerate(results_text):
            if text:  # Skip empty lines
                color = "Green" if "Successful" in text else "White"
                if "Crashes" in text and crash_count > 0:
                    color = "Red"
                elif "Out of Bounds" in text and out_of_bounds_count > 0:
                    color = "Orange"
                
                result_text = get_font(30).render(text, True, color)
                result_rect = result_text.get_rect(center=(640, y_offset + i * line_spacing))
                SCREEN.blit(result_text, result_rect)
        
        # Buttons
        MOUSE_POS = pygame.mouse.get_pos()
        
        BACK_BUTTON = Button(image=None, pos=(640, 550),
                            text_input="BACK TO MENU", font=get_font(40), 
                            base_color="White", hovering_color="Green")
        BACK_BUTTON.changeColor(MOUSE_POS)
        BACK_BUTTON.update(SCREEN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if BACK_BUTTON.checkForInput(MOUSE_POS):
                    return
        
        pygame.display.update()

def test_mode():
    """Enhanced test mode that loads and evaluates trained models."""
    if not pygame.get_init():
        pygame.init()
    
    # Model selection
    selected_model = model_selection_screen()
    if selected_model is None:
        return  # Go back to main menu
    
    # Configuration screen
    num_episodes = 10  # Default
    render = True  # Watching the agent play is the point of test mode
    random_target = True
    while True:
        SCREEN.fill("black")
        
        TITLE_TEXT = get_font(50).render("TEST CONFIGURATION", True, "#b68f40")
        TITLE_RECT = TITLE_TEXT.get_rect(center=(640, 100))
        SCREEN.blit(TITLE_TEXT, TITLE_RECT)
        
        MODEL_TEXT = get_font(30).render(f"Model: {selected_model['name']}", True, "White")
        MODEL_RECT = MODEL_TEXT.get_rect(center=(640, 180))
        SCREEN.blit(MODEL_TEXT, MODEL_RECT)
        
        EPISODES_TEXT = get_font(30).render(f"Episodes to test: {num_episodes}", True, "White")
        EPISODES_RECT = EPISODES_TEXT.get_rect(center=(640, 220))
        SCREEN.blit(EPISODES_TEXT, EPISODES_RECT)
        
        INFO_TEXT = get_font(25).render("Watch the AI agent play and see performance metrics", True, "Gray")
        INFO_RECT = INFO_TEXT.get_rect(center=(640, 260))
        SCREEN.blit(INFO_TEXT, INFO_RECT)
        
        MOUSE_POS = pygame.mouse.get_pos()
        
        # Episode count buttons
        DECREASE_BUTTON = Button(image=None, pos=(540, 320),
                                text_input="-", font=get_font(40), 
                                base_color="White", hovering_color="Red")
        INCREASE_BUTTON = Button(image=None, pos=(740, 320),
                                text_input="+", font=get_font(40), 
                                base_color="White", hovering_color="Green")
        
        # Option toggles
        RENDER_BUTTON = Button(image=None, pos=(640, 385),
                              text_input=f"Render episodes: {'ON' if render else 'OFF'}",
                              font=get_font(32),
                              base_color="Green" if render else "Red", hovering_color="White")
        TARGET_BUTTON = Button(image=None, pos=(640, 435),
                              text_input=f"Random target position: {'ON' if random_target else 'OFF'}",
                              font=get_font(32),
                              base_color="Green" if random_target else "Red", hovering_color="White")

        # Control buttons
        START_BUTTON = Button(image=None, pos=(640, 505),
                             text_input="START TEST", font=get_font(40),
                             base_color="Green", hovering_color="White")
        BACK_BUTTON = Button(image=None, pos=(640, 570),
                            text_input="BACK", font=get_font(40),
                            base_color="White", hovering_color="Green")

        for button in [DECREASE_BUTTON, INCREASE_BUTTON, RENDER_BUTTON, TARGET_BUTTON, START_BUTTON, BACK_BUTTON]:
            button.changeColor(MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if DECREASE_BUTTON.checkForInput(MOUSE_POS):
                    num_episodes = max(1, num_episodes - 1)
                elif INCREASE_BUTTON.checkForInput(MOUSE_POS):
                    num_episodes = min(20, num_episodes + 1)
                elif RENDER_BUTTON.checkForInput(MOUSE_POS):
                    render = not render
                elif TARGET_BUTTON.checkForInput(MOUSE_POS):
                    random_target = not random_target
                elif START_BUTTON.checkForInput(MOUSE_POS):
                    # Run the test
                    set_render_enabled(render)
                    set_random_target(random_target)
                    results = run_test_episodes(selected_model['path'], num_episodes, render)
                    if results is not None:
                        display_results(results)
                    return  # Return to main menu after test
                elif BACK_BUTTON.checkForInput(MOUSE_POS):
                    return  # Go back to main menu
        
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
