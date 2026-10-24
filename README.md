
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

Short answer: nothing in the reward mentioned velocity. `landed_ok` only checked
that the lander touched the pad, at any speed, and the shaping only knew about
distance. Trained a million steps on that reward the policy fell for 309 steps
and hit the ground at 2.524 px/frame -- a free fall takes 307 steps and arrives
at 2.53. It had not learned to fly at all, because falling scored exactly as
well. Zero soft landings in 300 episodes.

That was one of three separate problems. The second was that stalling in mid air
scored better than committing to a landing. The third, and the real ceiling, was
that **81% of episodes started with the pad further away than the lander could
physically reach** -- the booster covers 118px of ground during a free fall while
the mean spawn gap was 391px. That is not a policy failing to learn.

A 40-run ablation then narrowed down which reward change actually did the work:

| variant | soft landings | touchdown speed |
|---|---|---|
| full reward (control, 6 seeds) | 0.60 ±0.21 | 0.47 |
| minus the speed term in the shaping | **0.01** | **3.41** |
| the original reward, on the fixed task | **0.00** | **3.00** |

Touchdown speed splits in two with nothing in between: keep a speed term in the
shaping and the lander arrives at 0.47-0.70 px/frame, drop it and it arrives at
3.00-3.45, where a free fall is 3.44. One term does nearly all of the work, and
it has to be gated by distance or it paralyses the agent instead.

**[Read the full write-up -> `docs/reward-investigation.md`](docs/reward-investigation.md)**
-- all three problems with their measurements, a wrong turn that cost three
rounds of sweeps, the whole ablation table with permutation tests, the other
bugs found along the way, and why "100%" is the checkpoint rather than the
recipe.

To get the old task back: `SPAWN_MAX_GAP = None`, `ACCELERATION = 0.005`,
`fuel_penalty = 0.0`, `landing_speed_limit = math.inf`, `shaping = "banded"`.

## Where it ended up

Scored over 300 episodes as "landed on the pad at under 1.0 px/frame", the same
yardstick for every row regardless of the reward each was trained on.

| run | soft landings | mean touchdown speed |
|---|---|---|
| old reward, old hyperparameters, 1M | **0.0%** | 2.524 (free fall) |
| new reward, uncapped spawn, 1M | 2.7% | 1.100 |
| new reward, capped spawn, original booster, 3M | 38.7% | 0.792 |
| **new reward, capped spawn, 0.01 booster, 3M** | **100.0%** | 0.257 |

That last row is `pretrained/ppo_lander_3M.zip`, which TEST MODE lists without
you having to train anything. It holds up on unseen seeds (399/400 on a fresh
set), uses thrust on 57% of steps, and degrades gracefully: 96.7% at a 500px cap
and **73.0% on the original uncapped spawn range** it was never trained on.

**That 100% is the checkpoint, not the recipe.** Training this exact config six
times gives 0.60 ±0.21, ranging 0.23 to 0.86; the shipped model is the lucky
tail. Training is deterministic per seed on a given machine, but the same seed on
different hardware lands elsewhere in the range. Expect a good run to need a few
attempts, and treat any single run in this project as a hint rather than a
result -- the ablation needed six seeds per variant before anything but the
largest effect separated from noise.


## Running the experiments

    python tools/experiment.py --list
    python tools/experiment.py final --timesteps 3000000

    # the ablation above: one term at a time, several seeds each
    python tools/experiment.py abl_full --seed 0 --timesteps 3000000 --eval-every 500000
    python tools/ablation_report.py results/ablation

Each entry in `EXPERIMENTS` is one hypothesis, and scoring is deliberately
independent of the reward under test -- it reads the final game state -- so runs
with different rewards stay comparable.

    python -m unittest discover -s tests
