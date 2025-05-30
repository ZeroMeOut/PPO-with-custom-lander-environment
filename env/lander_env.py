import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import Any, Dict, List, Tuple, Union, Optional

from game_core.game_logic import run_game_frame, game_reset
from game_core.game_render import  quit_pygame


## From https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
class LanderEnvironment(gym.Env):
    metadata: Dict[str, Any] = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    action_space: spaces.Space
    observation_space: spaces.Space
    _np_random: Optional[np.random.Generator]

    def __init__(self) -> None:
        super(LanderEnvironment, self).__init__()
        ## Define action and observation space
        ## Action space: 0: left, 1: right, 2: up, 3: upleft, 4: upright, 5: do nothing
        self.action_space = spaces.Discrete(6)
        
        ## Observation space: 2D array with 6 elements
        self.observation_space = spaces.Box(low=np.array([-100, -100, -5.0, -5.0, 0, 0]),     
                                            high=np.array([1380, 650, 5.0, 5.0, 1280, 600]),
                                            shape=(6,), dtype=np.float64)
        
        self.seed()
        self.current_observation: Optional[np.ndarray] = None
    
    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        self._np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        result: Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]] = run_game_frame("training", action)
        if result is None:
            ## Provide default values if game_loop returns None
            observation: np.ndarray = np.zeros(6, dtype=np.float64)
            reward: float = 0.0
            terminated: bool = True
            truncated: bool = False
            info: Dict[str, Any] = {"message": "game_loop returned None, likely due to menu exit."}
        else:
            observation, reward, terminated, info = result 
            truncated = False
        self.current_observation = observation 
        return observation, reward, terminated, truncated, info

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        game_reset()
        self.current_observation = None
        result: Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]] = run_game_frame("training", action=3) ## Dummy action
        if result is None:
            observation: np.ndarray = np.zeros(6, dtype=np.float64)
            info: Dict[str, Any] = {"message": "game_loop returned None, likely due to menu exit."}
        else:
            observation, _, _, info = result
        return observation, info # Gym 0.26+ reset returns (observation, info)

    def render(self) -> Optional[Union[np.ndarray, bool]]:
        if self.render_mode == 'human':
            ## Pygame display.update() is already called in run_game_frame.
            return True
        elif self.render_mode == 'rgb_array':
            return self.current_observation
        
    def close(self) -> None:
        quit_pygame()