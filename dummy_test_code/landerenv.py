import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from landergame import game_loop
from typing import Any, Dict, List, Tuple, Union, Optional

WIDTH: int = 1280
HEIGHT: int = 720
N_CHANNELS: int = 3

## From https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
class LanderEnvironment(gym.Env):
	"""Custom Environment that follows gym interface"""

	metadata: Dict[str, Any] = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

	action_space: spaces.Space
	observation_space: spaces.Space
	_np_random: Optional[np.random.Generator]

	def __init__(self) -> None:
		super(LanderEnvironment, self).__init__()
		## Define action and observation space
		## Action space: 0: left, 1: right, 2: up, 3: do nothing
		self.action_space = spaces.Discrete(4)
		
		## Observation space: 3D array representing the screen pixels
		self.observation_space = spaces.Box(low=0, high=255,
											shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
		
		self.seed()
		self.current_observation: Optional[np.ndarray] = None

	def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
		self._np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
		result: Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]] = game_loop("training", action)
		if result is None:
			## Provide default values if game_loop returns None
			observation: np.ndarray = np.zeros((HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
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
		result: Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]] = game_loop("training", action=3) # Pass a dummy action, or no action at reset
		if result is None:
			observation: np.ndarray = np.zeros((HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
			info: Dict[str, Any] = {"message": "game_loop returned None, likely due to menu exit."}
		else:
			observation, _, _, info = result
		return observation, info # Gym 0.26+ reset returns (observation, info)
	
	def close(self) -> None:
		pass