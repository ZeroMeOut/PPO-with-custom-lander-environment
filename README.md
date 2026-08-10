
# PPO with custom lander environment
![Screenshot 2025-05-31 032530](https://github.com/user-attachments/assets/4eef797e-f016-47cf-af04-0f8900129b01)

Recently, I have been very interested in reinforcement learning, so I wanted to create a project around that. I built a custom lander game using pygame-ce and used the PPO implementation from stable-baseline-3.




## Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/ZeroMeOut/PPO-with-custom-lander-environment.git](https://github.com/ZeroMeOut/PPO-with-custom-lander-environment.git)
    cd PPO-with-custom-lander-environment
    ```

2.  **UV stuff:**
    ```bash
    uv sync
    ```


    
## Usage
    ```bash
    python main_menu.py
    ```
The manual mode for you to play the game, and the training to train the lander. The testing is for well, testing.




## Acknowledgements

 - [Sent's video](https://www.youtube.com/watch?v=uKnjGn8fF70&t=540s)
 - [Baraltech's main menu for pygame](https://github.com/baraltech/Menu-System-PyGamee)
 - [This very wonderful blog](https://antoinebrl.github.io/blog/rl-mars-lander/#reward-shaping)
 - [Phildecroos's lunar lander game](https://github.com/phildecroos/lunar_lander/)
 - [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/s/GGCzYxGnLp)
 


## Random Thoughts
First off, I was initially trying to use CNN to train this. But it's so damn stressful, I think the default encoder used by stable baseline might not be the greatest for this. I would try to make a custom one someday.

Secondly, it is annoyingly long to fine-tune and train. I took like 20-25 tries to get a decently looking ep_rew_mean graph, and even then I kept running into [value loss explosion](https://medium.com/@kaige.yang0110/in-training-ppo-how-to-balance-value-loss-and-policy-loss-cbf10d9d6b86) because of the hyperparameters or rewards I was using (thank you Antoinebrl, you saved my time a lot). And lord, the bugs, I don't even want to start with that.

Overall, this is still a learning process for me, and even with all the rambling I am doing rn I still find building this fun.

# Some updates
I am trying to make the code and agent better now that I got a Claude code subscription. The training is way faster now, and the value loss explosion was caused by a bug in my reward function that I didn't figure out at the time. The Lander still sucks tho, it doesn't reallly try to use it's boosters to slow down then navigate towards the target. I believe again it's a reward function problem that I will eventually figure out.

## Why it wasn't using its boosters

It was a reward problem, and then it was a physics problem. Both were measured
rather than guessed, with `tools/experiment.py`.

**Nothing ever asked it to brake.** `landed_ok` only checked that the lander
touched the pad, at any speed. A policy trained for 1M steps on the old reward
scored a 0.0% soft-landing rate over 300 episodes at a mean impact speed of
2.524 px/frame -- free fall terminal velocity is 2.53, and its mean episode
length was 309 against a 307 step free fall. It had not learned to fly at all,
because falling scored exactly as well. Touching the pad now needs an impact
speed under `landing_speed_limit`, and the potential charges for excess speed.

**Stalling paid better than landing.** Timing out cost nothing and the distance
shaping was already banked, so hovering near the pad beat committing to a
landing unless the agent could already land more than about half the time.
Hovering was also free, because thrust had no cost. There is a `fuel_penalty`
per frame of thrust now, which is what real Lunar Lander charges, and a small
`time_penalty`.

**The shaping walled off every stationary state.** Distance and speed entered
the potential independently, so a hovering lander that began to move paid for
the speed at once and only earned it back after closing ~233px. Inside that
radius moving was never locally worth it, and every configuration tried against
that potential converged on hovering until the clock ran out. The speed
allowance now grows with distance (`approach_scale`): cruise fast, arrive slow.
Fixing this dropped timeouts from 300/300 to 44/300.

**Most episodes could not be flown.** This turned out to be the real ceiling.
The booster accelerates at 0.005 px/frame², so an accelerate-then-decelerate
crossing covers only `0.00125 * T²` px -- 118px during the 307 frame free fall.
The lander and the pad were drawn independently from x in [20, 1200], a mean gap
of 391px and a p90 of 804px, so **81% of episodes started with the pad further
away than the lander could reach**. The trained agent scored 12% inside 200px
and 0.0% at every distance beyond it. That is not a policy failing to learn. So
`SPAWN_MAX_GAP` caps how far the pad can be placed from the lander, and
`ACCELERATION` is a named constant you can raise instead.

Set `SPAWN_MAX_GAP = None`, `ACCELERATION = 0.005` and `fuel_penalty = 0.0` to
get the old task back.

## Where it ended up

Scored over 300 episodes as "landed on the pad at under 1.0 px/frame", which is
the same yardstick for every row regardless of the reward each was trained on.

| run | soft landings | mean touchdown speed | how episodes ended |
|---|---|---|---|
| old reward, old hyperparameters, 1M | **0.0%** | 2.524 (free fall) | 256 crashed, 34 touched the pad, 10 out of bounds |
| new reward, uncapped spawn, 1M | 2.7% | 1.100 | 175 crashed, 44 timed out, 42 reached the pad |
| new reward, capped spawn, original booster, 3M | 38.7% | 0.792 | 116 landed, 49 too fast, 104 timed out |
| **new reward, capped spawn, 0.01 booster, 3M** | **100.0%** | 0.257 | 300 landed |

The last row is `pretrained/ppo_lander_3M.zip`, which TEST MODE lists without
you having to train anything. It repeats on a second seed (300/300 again), holds
up on 400 unseen seeds (400/400), and it does use the boosters: thrust on 57% of
steps. It also degrades in the direction you would hope rather than falling over
-- the same model scores 96.7% at a 500px cap and **73.0% on the original
uncapped spawn range** it was never trained on.

Two things worth knowing before trusting any of the middle rows: single seed
runs at 1M steps disagreed with each other by more than several of the effects
being measured, and every one of the early configurations converged on hovering,
which looked like a hyperparameter problem for a long time and was not.

## Running the experiments

    python tools/experiment.py --list
    python tools/experiment.py final --timesteps 3000000

Each entry in `EXPERIMENTS` is one hypothesis, and scoring is deliberately
independent of the reward under test -- it reads the final game state -- so runs
with different rewards stay comparable. Single-seed runs at 1M steps disagreed
with each other by more than some of the effects being measured, so treat one
run as a hint and not a result.

    python -m unittest discover -s tests
