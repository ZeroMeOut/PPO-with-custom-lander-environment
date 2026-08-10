"""Headless train-and-score harness for reward and hyperparameter sweeps.

The menu is the way to train a model you want to keep. This is the way to answer
"is this config better than that one", which needs no window, no frame rate and
a number at the end.

    python tools/experiment.py --list
    python tools/experiment.py baseline_new tuned --timesteps 1000000

Scoring is deliberately independent of the reward being tested: it reads the
final game state, so a run that changes the reward can still be compared against
every other run on the same axis.
"""
import argparse
import collections
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import game_core.game_logic as game_logic
from game_core.game_logic import (RewardConfig, set_reward_config, set_random_target,
                                  set_spawn_max_gap, set_acceleration)
from env.lander_env import LanderEnvironment

## Impact speed that counts as a soft landing when scoring, held fixed across
## every experiment even when the config under test grades landings differently.
SCORE_SPEED_LIMIT = 1.0

## PPO settings shared by every experiment unless a config overrides them.
BASE_PPO = dict(
    policy="MlpPolicy",
    n_envs=8,
    n_steps=256,          # per env, so 2048 transitions per rollout
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    vf_coef=0.5,
    clip_range=0.2,
)


def experiment(reward=None, max_episode_steps=1000, random_target=True,
               spawn_max_gap=350, acceleration=0.005, **ppo):
    return {"reward": replace(RewardConfig(), **(reward or {})),
            "ppo": {**BASE_PPO, **ppo},
            "max_episode_steps": max_episode_steps,
            "random_target": random_target,
            "spawn_max_gap": spawn_max_gap,
            "acceleration": acceleration}


## Each entry is one hypothesis. Comments say what it is meant to isolate.
##
## Everything up to round five carries spawn_max_gap=None, the uncapped spawn
## geometry those runs were measured under, so their recorded numbers stay
## reproducible after the default changed.
EXPERIMENTS = {
    ## The repo's own PPO settings, on the new reward. Separates "the reward was
    ## wrong" from "the hyperparameters were wrong".
    "old_hparams": experiment(ent_coef=0.05, spawn_max_gap=None),

    ## Same reward, ent_coef back to the rl-zoo value the comments cite.
    "tuned": experiment(spawn_max_gap=None),

    ## Does the entropy bonus matter as much as it looks like it should?
    "ent_003": experiment(ent_coef=0.003, spawn_max_gap=None),
    "ent_02": experiment(ent_coef=0.02, spawn_max_gap=None),

    ## Longer rollouts: the credit assignment here spans a whole descent.
    "rollout_4096": experiment(n_steps=512, batch_size=128, spawn_max_gap=None),

    ## Is the shaping strong enough to guide the descent, or is it drowning the
    ## terminal signal?
    "shaping_weak": experiment(reward=dict(w_dist=50.0, w_speed=25.0), spawn_max_gap=None),
    "shaping_strong": experiment(reward=dict(w_dist=200.0, w_speed=100.0), spawn_max_gap=None),

    ## Does braking need to be worth more than approaching?
    "speed_heavy": experiment(reward=dict(w_speed=150.0), spawn_max_gap=None),

    ## Is the per-step cost doing the anti-stalling work, or is an explicit
    ## timeout penalty needed on top?
    "no_time_penalty": experiment(reward=dict(time_penalty=0.0), spawn_max_gap=None),
    "timeout_penalty": experiment(reward=dict(timeout_penalty=100.0), spawn_max_gap=None),

    ## Is a 1.0 px/frame touchdown too strict to ever stumble into?
    "loose_limit": experiment(reward=dict(landing_speed_limit=1.5), spawn_max_gap=None),
    "strict_limit": experiment(reward=dict(landing_speed_limit=0.6), spawn_max_gap=None),

    ## Shorter horizon: 0.999 looks far-sighted for ~400 step episodes.
    "gamma_99": experiment(gamma=0.99, spawn_max_gap=None),

    ## Added after the first round. "tuned" learned to brake but then dawdled:
    ## 79% of its episodes ran out the clock at a mean length of 959/1000, so it
    ## was descending too slowly to ever arrive. These price delay harder.
    "time_15": experiment(reward=dict(time_penalty=0.15), spawn_max_gap=None),
    "time_30": experiment(reward=dict(time_penalty=0.30), spawn_max_gap=None),
    "time_15_timeout": experiment(reward=dict(time_penalty=0.15, timeout_penalty=100.0), spawn_max_gap=None),

    ## The other reading of the same result: the potential scores "slow and far
    ## away" as good, because distance and speed are independent terms. Cutting
    ## the speed weight should stop rewarding a stall in mid air.
    "speed_light": experiment(reward=dict(w_speed=20.0, time_penalty=0.15), spawn_max_gap=None),

    ## Round three. Raising time_penalty traded timeouts for crashes rather than
    ## for landings, because it rushes the descent as much as it forbids the
    ## stall. Fuel is the targeted version of the same pressure: hovering burns
    ## every frame, falling burns nothing.
    ## A sweep of these against fixed policies put the useful range at 0.05-0.1:
    ## below it hovering stays viable, above it braking costs so much that a
    ## dead fall scores as well as a controlled descent.
    "fuel_002": experiment(reward=dict(fuel_penalty=0.02), spawn_max_gap=None),
    "fuel_005": experiment(reward=dict(fuel_penalty=0.05), spawn_max_gap=None),
    "fuel_01": experiment(reward=dict(fuel_penalty=0.1), spawn_max_gap=None),

    ## A shorter clock is the blunt way to force the same commitment. Worth
    ## knowing whether it is enough on its own.
    "short_episode": experiment(reward=dict(fuel_penalty=0.0), max_episode_steps=600, spawn_max_gap=None),
    "fuel_005_short": experiment(reward=dict(fuel_penalty=0.05), max_episode_steps=600, spawn_max_gap=None),

    ## Round four, and the one that mattered. Everything above hovered because
    ## the potential charged for speed everywhere, which walls off every
    ## stationary state. approach_scale makes the speed allowance grow with
    ## distance, so closing on the pad is worth doing from anywhere.
    "approach": experiment(spawn_max_gap=None),
    "approach_nofuel": experiment(reward=dict(fuel_penalty=0.0), spawn_max_gap=None),
    "approach_absolute": experiment(reward=dict(approach_scale=0.0), spawn_max_gap=None),
    ## Same reward against the easier task the menu already offers, to separate
    ## "the reward is wrong" from "a random pad is hard to hit".
    "approach_static": experiment(random_target=False, spawn_max_gap=None),
    ## Tighter and looser approach allowances.
    "approach_120": experiment(reward=dict(approach_scale=120.0), spawn_max_gap=None),
    "approach_300": experiment(reward=dict(approach_scale=300.0), spawn_max_gap=None),

    ## Round five: the task itself. Measuring reach against the spawn geometry
    ## showed 81% of episodes start with the pad further away than the lander can
    ## fly to, and the trained "approach" model scored 0.0% on every one of them.
    ## These cap the gap, or give the booster the authority the old range assumed.
    "reachable": experiment(spawn_max_gap=350),
    "reachable_250": experiment(spawn_max_gap=250),
    "reachable_500": experiment(spawn_max_gap=500),
    "thrust_2x": experiment(spawn_max_gap=None, acceleration=0.01),

    ## Round six. Capping the gap and doubling the booster each helped on their
    ## own but neither made the task comfortably flyable, and single seed runs at
    ## 1M steps disagreed with each other by more than the effects being measured
    ## (reachable 0.0% against reachable_500 6.0% on the same reward). Both fixes
    ## together, trained long enough to read: at 0.01 px/frame^2 a free fall lasts
    ## ~289 frames and covers ~209px of ground, so a 250px cap is about the widest
    ## gap the lander can close without paying to hover.
    "final": experiment(spawn_max_gap=250, acceleration=0.01),
    "final_wide": experiment(spawn_max_gap=350, acceleration=0.01),

    ## Does the capped geometry alone carry it, on the booster the game shipped
    ## with? This decides whether ACCELERATION has to change at all.
    "final_weak_booster": experiment(spawn_max_gap=250, acceleration=0.005),
}


def evaluate(model, n_episodes=300, seed0=100_000, max_episode_steps=1000):
    """Roll out a trained policy and score it on outcomes, not on its own reward."""
    env = LanderEnvironment(max_episode_steps=max_episode_steps)
    outcomes = collections.Counter()
    on_pad = soft = 0
    impacts, returns, lengths = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            done = terminated or truncated
        gs = env.game_state
        impact = math.hypot(gs.player.x_acceleration, gs.player.y_acceleration)
        if gs.player.y > game_logic.GROUND_Y and gs.player.collided_with(gs.target):
            on_pad += 1
            impacts.append(impact)
            if impact <= SCORE_SPEED_LIMIT:
                soft += 1
        outcomes[info.get("status", "unknown")] += 1
        returns.append(total)
        lengths.append(steps)
    return {
        "soft_landing_rate": round(soft / n_episodes, 4),
        "on_pad_rate": round(on_pad / n_episodes, 4),
        "mean_impact_speed": round(sum(impacts) / len(impacts), 3) if impacts else None,
        "mean_return": round(sum(returns) / n_episodes, 2),
        "mean_length": round(sum(lengths) / n_episodes, 1),
        "outcomes": dict(outcomes),
    }


def run(name, timesteps, seed, eval_episodes, results_dir):
    config = EXPERIMENTS[name]
    set_reward_config(config["reward"])
    set_random_target(config["random_target"])
    set_spawn_max_gap(config["spawn_max_gap"])
    set_acceleration(config["acceleration"])

    ppo_kwargs = dict(config["ppo"])
    n_envs = ppo_kwargs.pop("n_envs")
    policy = ppo_kwargs.pop("policy")
    max_episode_steps = config["max_episode_steps"]

    env = make_vec_env(LanderEnvironment, n_envs=n_envs, seed=seed,
                       env_kwargs={"max_episode_steps": max_episode_steps})
    model = PPO(policy, env, verbose=0, device="cpu", seed=seed, **ppo_kwargs)

    started = time.time()
    model.learn(total_timesteps=timesteps)
    train_seconds = time.time() - started

    result = {
        "name": name,
        "seed": seed,
        "timesteps": timesteps,
        "train_seconds": round(train_seconds, 1),
        "reward": asdict(config["reward"]),
        "ppo": config["ppo"],
        "max_episode_steps": max_episode_steps,
        "random_target": config["random_target"],
        "spawn_max_gap": config["spawn_max_gap"],
        "acceleration": config["acceleration"],
        **evaluate(model, n_episodes=eval_episodes, max_episode_steps=max_episode_steps),
    }
    env.close()

    os.makedirs(results_dir, exist_ok=True)
    model.save(os.path.join(results_dir, f"{name}_seed{seed}"))
    with open(os.path.join(results_dir, f"{name}_seed{seed}.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="experiment names to run")
    parser.add_argument("--list", action="store_true", help="list experiment names")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--threads", type=int, default=2,
                        help="torch threads; 2 measured fastest on 4 cores")
    args = parser.parse_args()

    if args.list or not args.names:
        for name in EXPERIMENTS:
            print(name)
        return

    torch.set_num_threads(args.threads)
    for name in args.names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; --list to see them all")
        result = run(name, args.timesteps, args.seed, args.eval_episodes, args.results_dir)
        print(json.dumps(result))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
