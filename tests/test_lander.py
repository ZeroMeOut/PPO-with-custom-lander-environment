"""Regression tests for the environment and the reward.

Plain unittest so they run with no extra dependency:

    python -m unittest discover -s tests

Most of these exist because the behaviour they check was once wrong in a way
that was invisible from the training curves. A reward bug does not raise; it
just quietly teaches the agent something else, so it needs a test more than
ordinary code does.
"""
import math
import os
import unittest
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import game_core.game_logic as game_logic
from game_core.game_logic import (GameState, RewardConfig, calculate_reward_and_done,
                                  phi, set_reward_config, set_random_target,
                                  set_spawn_max_gap)
from env.lander_env import LanderEnvironment


class RewardConfigCase(unittest.TestCase):
    """Restores the module globals the game keeps its settings in."""

    def setUp(self):
        self._reward = game_logic.REWARD
        self._gap = game_logic.SPAWN_MAX_GAP
        self._random = game_logic.RANDOM_TARGET
        self.addCleanup(lambda: set_reward_config(self._reward))
        self.addCleanup(lambda: set_spawn_max_gap(self._gap))
        self.addCleanup(lambda: set_random_target(self._random))

    def touchdown(self, gs, on_pad=True, vx=0.0, vy=0.0):
        """Put the lander on the ground, on or off the pad, at a given speed."""
        gs.target.x = 600
        gs.target.rect.topleft = (600, game_logic.GROUND_Y)
        gs.player.x = 600 if on_pad else 100
        gs.player.y = game_logic.GROUND_Y + 1
        gs.player.rect.topleft = (int(gs.player.x), int(gs.player.y))
        gs.player.x_acceleration = vx
        gs.player.y_acceleration = vy
        return calculate_reward_and_done(gs, drawing=False)


class TestLandingSpeedGate(RewardConfigCase):
    def test_slow_touchdown_on_pad_is_a_landing(self):
        gs = GameState(seed=0)
        _, done, info = self.touchdown(gs, on_pad=True, vy=0.3)
        self.assertTrue(done)
        self.assertEqual(info["status"], "landed_ok")

    def test_fast_touchdown_on_pad_is_not_a_landing(self):
        """The whole point. A free fall onto the pad used to score a clean
        success, so nothing ever asked the agent to brake."""
        gs = GameState(seed=0)
        _, done, info = self.touchdown(gs, on_pad=True, vy=2.5)
        self.assertTrue(done)
        self.assertEqual(info["status"], "too_fast")

    def test_horizontal_speed_counts_towards_the_limit(self):
        gs = GameState(seed=0)
        _, _, info = self.touchdown(gs, on_pad=True, vx=2.5, vy=0.0)
        self.assertEqual(info["status"], "too_fast")

    def test_landing_pays_more_than_arriving_too_fast(self):
        slow, _, _ = self.touchdown(GameState(seed=0), on_pad=True, vy=0.2)
        fast, _, _ = self.touchdown(GameState(seed=0), on_pad=True, vy=2.5)
        miss, _, _ = self.touchdown(GameState(seed=0), on_pad=False, vy=2.5)
        self.assertGreater(slow, fast)
        self.assertGreater(fast, miss)


class TestShaping(RewardConfigCase):
    def test_shaping_is_path_independent(self):
        """The shaping sums to phi(end) - phi(start) whatever route is taken, so
        no trajectory can farm it by wandering back and forth."""
        set_reward_config(replace(RewardConfig(), time_penalty=0.0, fuel_penalty=0.0))
        gs = GameState(seed=3)
        start_phi = phi(gs)
        total = 0.0
        for step in range(200):
            gs.player.x += 3 if (step // 20) % 2 == 0 else -2
            gs.player.y += 1.5
            gs.player.rect.topleft = (int(gs.player.x), int(gs.player.y))
            reward, done, _ = calculate_reward_and_done(gs, drawing=False)
            total += reward
            if done:
                break
        self.assertAlmostEqual(total, phi(gs) - start_phi, places=6)

    def test_moving_towards_the_pad_is_never_punished_at_range(self):
        """The barrier that made every early config hover: with a flat speed
        penalty, a stationary lander lost potential the moment it moved."""
        gs = GameState(seed=1)
        gs.target.x, gs.target.y = 600, game_logic.GROUND_Y
        for distance_px in (600, 500, 400, 300):
            gs.player.x = 600
            gs.player.y = game_logic.GROUND_Y - distance_px
            gs.player.x_acceleration = 0.0

            gs.player.y_acceleration = 0.0
            still = phi(gs)
            gs.player.y += 1.0
            gs.player.y_acceleration = 1.0
            moving = phi(gs)
            self.assertGreaterEqual(
                moving, still,
                f"starting to move at {distance_px}px lost potential")

    def test_speed_is_charged_close_to_the_pad(self):
        """The allowance shrinks with distance, so arriving fast still costs."""
        gs = GameState(seed=1)
        gs.target.x, gs.target.y = 600, game_logic.GROUND_Y
        gs.player.x, gs.player.y = 600, game_logic.GROUND_Y - 20

        gs.player.y_acceleration = 0.0
        still = phi(gs)
        gs.player.y_acceleration = 2.0
        self.assertLess(phi(gs), still)


class TestIncentives(RewardConfigCase):
    def test_hovering_to_timeout_scores_worse_than_landing(self):
        """Under the original reward this comparison came out the other way:
        stalling near the pad beat committing to a landing."""
        gs = GameState(seed=5)
        gs.target.x = 600
        gs.target.rect.topleft = (600, game_logic.GROUND_Y)
        gs.player.x, gs.player.y = 600, 200
        gs.player.x_acceleration = gs.player.y_acceleration = 0.0
        gs.prev_phi = phi(gs)

        hovering = 0.0
        for _ in range(800):
            gs.is_thrusting = True     # holding altitude means burning every frame
            reward, _, _ = calculate_reward_and_done(gs, drawing=False)
            hovering += reward

        landing_gs = GameState(seed=5)
        landing, _, info = self.touchdown(landing_gs, on_pad=True, vy=0.3)
        self.assertEqual(info["status"], "landed_ok")
        self.assertGreater(landing, hovering)

    def test_thrusting_costs_fuel(self):
        gs = GameState(seed=7)
        gs.prev_phi = phi(gs)
        gs.is_thrusting = False
        coasting, _, _ = calculate_reward_and_done(gs, drawing=False)

        gs = GameState(seed=7)
        gs.prev_phi = phi(gs)
        gs.is_thrusting = True
        burning, _, _ = calculate_reward_and_done(gs, drawing=False)

        self.assertAlmostEqual(coasting - burning, game_logic.REWARD.fuel_penalty, places=6)


class TestGameState(RewardConfigCase):
    def test_reset_restores_the_potential_baseline(self):
        """Left stale, the first step of an episode is scored against the
        previous episode's final state."""
        gs = GameState(seed=11)
        gs.player.y = 400
        calculate_reward_and_done(gs, drawing=False)
        gs.reset()
        self.assertAlmostEqual(gs.prev_phi, phi(gs), places=9)

    def test_first_step_of_an_episode_is_small(self):
        set_reward_config(replace(RewardConfig(), time_penalty=0.0, fuel_penalty=0.0))
        gs = GameState(seed=13)
        gs.reset()
        reward, _, _ = calculate_reward_and_done(gs, drawing=False)
        self.assertLess(abs(reward), 1.0)

    def test_spawn_gap_is_respected(self):
        set_spawn_max_gap(350)
        set_random_target(True)
        gs = GameState(seed=0)
        for seed in range(300):
            gs.seed(seed)
            gs.reset()
            self.assertLessEqual(abs(gs.player.x - gs.target.x), 350)

    def test_spawn_gap_none_restores_the_full_range(self):
        set_spawn_max_gap(None)
        set_random_target(True)
        gs = GameState(seed=0)
        gaps = []
        for seed in range(400):
            gs.seed(seed)
            gs.reset()
            gaps.append(abs(gs.player.x - gs.target.x))
        self.assertGreater(max(gaps), 350)

    def test_seeding_is_reproducible(self):
        a, b = GameState(seed=42), GameState(seed=42)
        a.reset(); b.reset()
        self.assertEqual((a.player.x, a.target.x), (b.player.x, b.target.x))


class TestEnvironment(RewardConfigCase):
    def test_observations_stay_inside_the_declared_box(self):
        env = LanderEnvironment()
        obs, _ = env.reset(seed=0)
        space = env.observation_space
        self.assertTrue(space.contains(obs))
        for episode in range(5):
            obs, _ = env.reset(seed=episode)
            done = False
            while not done:
                obs, _, terminated, truncated, _ = env.step(episode % 6)
                self.assertTrue(space.contains(obs), f"escaped the box: {obs}")
                done = terminated or truncated

    def test_reset_returns_the_untouched_initial_state(self):
        """reset used to step a frame with a hardcoded action, so every episode
        opened with a thrust the agent never chose."""
        env = LanderEnvironment()
        obs, _ = env.reset(seed=17)
        np.testing.assert_allclose(obs, env.game_state.get_observation())

    def test_truncation_is_flagged_for_bootstrapping(self):
        """Without this key SB3 scores every timeout as a real terminal worth
        zero, whatever the lander had achieved by then."""
        env = LanderEnvironment(max_episode_steps=5)
        env.reset(seed=0)
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(2)  # thrust, never lands
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["status"], "timeout")
        self.assertTrue(info.get("TimeLimit.truncated"))

    def test_terminal_observation_is_the_terminal_state(self):
        """Resetting inside the reward handed back the *next* episode's spawn
        point as the terminal observation."""
        env = LanderEnvironment()
        env.reset(seed=2)
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(5)
            done = terminated or truncated
        np.testing.assert_allclose(obs, env.game_state.get_observation())
        self.assertGreater(env.game_state.player.y, game_logic.GROUND_Y)

    def test_episode_ends_within_the_step_budget(self):
        env = LanderEnvironment(max_episode_steps=50)
        env.reset(seed=0)
        for step in range(50):
            _, _, terminated, truncated, _ = env.step(5)
            if terminated or truncated:
                break
        self.assertTrue(terminated or truncated)


if __name__ == "__main__":
    unittest.main()
