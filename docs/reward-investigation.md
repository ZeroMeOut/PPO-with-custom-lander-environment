# Why the lander wasn't using its boosters

A record of what was wrong, how each thing was measured, and which part of the
reward actually mattered. Everything here is reproducible with
`tools/experiment.py`; the per-run numbers are in `results/` and
`results/ablation/`.

---

## The short version

The lander never braked because **nothing in the reward mentioned velocity.**
`landed_ok` only checked that it touched the pad, at any speed, and the shaping
only knew about distance. A policy trained a million steps on that reward was
statistically indistinguishable from one doing nothing at all.

That was the first of three separate problems. The second was that stalling in
mid air scored better than committing to a landing. The third, and the real
ceiling, was that **81% of episodes started with the pad further away than the
lander could physically fly to** — not a learning failure, a task that mostly
could not be flown.

Fixing all three takes the lander from 0% to a trained checkpoint that lands
399 times out of 400. A 40-run ablation then narrowed down *which* reward change
did the work: a speed term in the shaping, gated by distance. Almost everything
else turned out to be within noise.

---

## How things were measured

One yardstick, used for every run in this document regardless of what reward it
was trained on:

> **soft landing** = touched down on the pad with a total speed at or under
> 1.0 px/frame.

It is read off the final game state, not from the reward, so a run that changes
the reward is still comparable with every other run. This matters more than it
sounds: several rewards here report success for touchdowns the yardstick counts
as crashes, which is exactly the bug.

Two secondary metrics turned out to carry more information than the headline:

- **touchdown speed** — says whether the lander is flying or falling. Free fall
  arrives at 2.53 px/frame at the original booster strength, 3.44 at the
  current one.
- **timeout rate** — says whether it is stalling instead of landing.

Both are far steadier across seeds than the landing rate, so they are what
identify a missing reward term.

---

## Problem 1: nothing asked it to brake

`landed_ok` fired on `player.y > GROUND_Y and collided_with(target)`. No speed
check. The shaping was a distance band counter with no velocity term. So
velocity appeared nowhere in the reward at all.

Measured consequence, 1M steps on that reward, 300 episodes:

| | value | for comparison |
|---|---|---|
| soft landing rate | **0.000** | — |
| mean touchdown speed | **2.524** px/frame | free fall is 2.53 |
| mean episode length | **309** steps | free fall takes 307 |

After a million steps the policy fell for the same duration and hit the ground
at the same speed as doing nothing. It had learned a little horizontal steering
and nothing else, because falling scored exactly as well as flying. Zero soft
landings in 300 episodes, while its own reward reported 34 successes.

Verified directly with fixed hand-written policies: braking was measurably
*worse* than not braking (+16.8 vs +19.0 mean return). The agent was correct.

**Fix:** touching the pad now requires an impact speed under
`landing_speed_limit`, and arriving too fast is graded separately (`too_fast`,
−50) from missing entirely (`crashed`, −100) — a worse outcome than landing, a
better one than a total miss, which gives the policy a route in.

### The banded shaping had other problems too

`phi` paid a flat ±10 for crossing a distance band and nothing in between:

- Bands measured **54–125px wide** (mean 70), so the entire final approach sat
  inside band 0 with no gradient at all — no signal precisely where precision
  matters.
- Band width came from the spawn distance, so identical physical progress paid
  differently from episode to episode, about a 2.3× spread.
- `prev_band` snapped on any change, so crossing five bands in one step paid
  +10 where crossing them one at a time paid +50.
- It is not a potential despite the name: it mutates state and its return is
  used as the reward directly, not as a difference.

**Fix:** a continuous potential, applied as `phi(s') - phi(s)`.

---

## Problem 2: stalling paid better than landing

Timing out cost nothing, the distance shaping was already banked, and thrust
was free. So hovering near the pad kept the shaping and risked nothing.

Measured with fixed policies on the original reward:

| policy | mean return | outcome |
|---|---|---|
| hover to timeout | **+58.3** | 300/300 timeout |
| try to land | +40.8 | 49 landed, 101 crashed |

Hovering beat attempting a landing. Break-even was around 50% landing accuracy —
below that, doing nothing near the pad strictly dominated.

**Fix:** a `fuel_penalty` charged per frame of thrust, which is how the real
Lunar Lander prices it, plus a small `time_penalty`. Fuel is the targeted
version of the pressure: holding altitude burns every frame, while falling and
braking late is nearly free. A plain time penalty prices *all* delay equally, so
raising it enough to forbid a stall also rushes the descent — at
`time_penalty=0.30` the agent learned to fly *up* and end the episode early,
280 of 300 episodes ending in `flown_too_high`.

A sweep against fixed policies put fuel's useful range at 0.05–0.1. Past about
0.2 the bill for braking grows until falling like a stone scores as well as
flying.

---

## Problem 3: most episodes could not be flown

This was the real ceiling, and it is geometric rather than algorithmic.

The booster accelerates at 0.005 px/frame², so an accelerate-then-decelerate
crossing covers `0.00125 × T²` px:

| frames | horizontal reach |
|---|---|
| 307 (a free fall) | **118 px** |
| 500 | 312 px |
| 1000 (the whole episode) | 1250 px |

But the lander and the pad were drawn independently from x ∈ [20, 1200]: mean
gap **391px**, median 346, p90 **804**, max 1170. So **80.9% of episodes began
with the pad beyond what the lander could reach** without hovering to buy time.

The trained agent confirms it exactly:

| initial horizontal gap | soft landing rate |
|---|---|
| 0–200 px | 0.120 |
| 200–400 px | 0.007 |
| 400–600 px | 0.000 |
| 600–800 px | 0.000 |
| 800–1000 px | 0.000 |
| 1000–1200 px | 0.000 |

12% inside 200px, and essentially zero beyond it. Hovering was not only a reward
artefact — it was frequently the best available play.

**Fix:** `SPAWN_MAX_GAP` caps how far the pad can be placed from the lander
(250px), keeping both ends drawn from the same range so the old left-bias stays
fixed. `ACCELERATION` became a named constant and was raised to 0.01.

Raising the booster is the one change that alters how the game feels to play, so
it was tested rather than assumed: on the capped geometry at 3M steps, 0.005
scored 38.7% against 100% for 0.01. Both of those are single runs, though, and
the variance section below says a single run here is worth about as much as a
hint — the honest version of that comparison needs several seeds each and has
not been done. What is clear is that 0.005 still gives a lander that flies,
38.7% instead of 0.0%, and that the reward fixes are what got it there.

---

## A wrong turn worth recording: the potential wall

This one was self-inflicted during the rewrite, not a pre-existing bug, and it
cost three rounds of sweeps.

Replacing the bands with `phi = -(w_dist·d/DIST_SCALE + w_speed·v/SPEED_SCALE)`
made distance and speed independent penalties. But **speed is the derivative of
distance**: closing the gap *requires* carrying the thing being penalised. The
balance between the two weights decides how badly that hurts.

The useful way to express that balance is a **payback time** — how many frames of
travel it takes to earn back the cost of carrying speed:

```
payback_frames = (w_speed / SPEED_SCALE) / (w_dist / DIST_SCALE)
```

Acquiring speed `v` costs `(w_speed/V)·v` once; travelling at `v` earns
`(w_dist/D)·v` every frame. The ratio is a time, independent of `v`.

| weights | payback |
|---|---|
| `-(1.0·d + 8.0·v)` (an earlier commit) | **8 frames** |
| `w_dist=100/1400, w_speed=50/3` (the rewrite) | **233 frames** |

A landing takes ~500 frames, so the rewrite demanded the agent commit to
travelling for nearly half an episode before the shaping stopped punishing it.
Traced along a textbook approach from 400px, the potential falls from −28.6 to
−47.6 before recovering: hundreds of steps of negative reward for flying
correctly, with the refund arriving only if the approach finishes. And inside
233px of the pad, no route to it paid for the speed it needed at all.

Every configuration tried against that potential converged on hovering —
`old_hparams` 300/300 timeouts, `shaping_strong` 298/300, `fuel_005` 299/300 —
which looked like a hyperparameter problem for a long time and was not one.

The form was never the problem. Normalising distance by 1400 while normalising
speed by only 3 inflated the speed term about 29-fold.

**Fix:** let the speed allowance grow with distance, and charge only the excess:

```
v_ref(d) = clamp(d / approach_scale, approach_v_min, approach_v_max)
phi = -(w_dist·d/DIST_SCALE + w_speed·max(0, v - v_ref(d))/SPEED_SCALE)
```

This states the thing you actually want — travel fast when far, arrive slow when
close. The valley drops from 19.0 deep to 2.4, and the residual is the genuine
cost of arriving too hot. Timeouts went from 300/300 to 44/300 on the next run.

### How to normalise a potential, in two steps

1. Divide each raw quantity by its physical range so both terms are ~[0,1].
   This is the easy half and takes |Φ| from ~670 to ~40.
2. **Set the ratio by payback time, not by taste.** Normalising each term
   separately says nothing about how they trade against each other. Target a
   payback that is a small fraction of an episode, 10–30 frames rather than 233:
   `w_speed = payback_frames · w_dist · SPEED_SCALE / DIST_SCALE`, which here
   gives `w_speed ≈ 5`, not 50.

Also worth knowing: the speed term in Φ is not what teaches braking — see the
ablation. The terminal gate defines success; the shaping only needs to avoid
fighting travel, which is why it can be small.

---

## Which part of the reward actually did the work

Everything above changed several things at once, so "the reward got better" was
never attributed to a part of it. This is the experiment that attributes it.

**40 training runs of 3M steps** (120M steps total). The task is held fixed at
the setting that is actually flyable, so the only thing varying between runs is
the named reward term. `p` is an exact permutation test against the control; at
2 seeds vs 6 the smallest reachable value is 0.036, so `p=0.036` means "as
separated as this sample size can show", not "weak".

| variant | n | soft landings | p | touchdown speed | p | timeout rate | p |
|---|---|---|---|---|---|---|---|
| FULL REWARD (control) | 6 | 0.60 ±0.21 | | 0.47 ±0.16 | | 0.02 ±0.05 | |
| minus time penalty | 4 | 0.55 ±0.38 | 0.84 | 0.62 ±0.14 | 0.25 | 0.25 ±0.43 | 0.40 |
| minus landing speed gate | 6 | 0.58 ±0.36 | 0.94 | 0.70 ±0.40 | 0.27 | 0.02 ±0.03 | 0.73 |
| minus fuel penalty | 6 | 0.45 ±0.36 | 0.43 | 0.54 ±0.16 | 0.50 | 0.33 ±0.47 | 0.38 |
| minus fuel AND time | 2 | 0.22 ±0.21 | 0.14 | 0.23 ±0.09 | 0.14 | **0.49 ±0.02** | **0.036** |
| minus approach gating | 6 | 0.22 ±0.36 | 0.08 | 0.60 ±0.40 | 0.62 | **0.60 ±0.45** | 0.06 |
| minus speed term in phi | 3 | **0.01 ±0.01** | **0.012** | **3.41 ±0.09** | **0.036** | 0.27 ±0.38 | 0.33 |
| banded shaping instead | 2 | **0.00** | **0.036** | **3.39 ±0.06** | **0.036** | 0.00 | 0.79 |
| no shaping at all | 2 | **0.00** | **0.036** | **3.45** | 0.14 | 0.30 ±0.30 | 0.25 |
| THE ORIGINAL REWARD | 3 | **0.00** | **0.012** | **3.00 ±0.22** | **0.012** | 0.00 | 0.75 |

### Read the touchdown-speed column first

It splits cleanly in two, with nothing in between:

- every variant that **keeps** a speed term in the shaping arrives at
  **0.47–0.70** px/frame
- every variant that **drops** it arrives at **3.00–3.45** px/frame, where a
  free fall touches down at 3.44

**One term does almost all of the work: speed in the shaping.** Take it out and
the lander stops braking entirely, whatever else is present. The three other
rows that score zero — bands, no shaping, the original reward — are not separate
failures; they are three different ways of not having a speed term.

### The cleanest single result

The original reward — banded shaping, no speed gate, no fuel, no time cost — run
on the task that is now flyable:

```
THE ORIGINAL REWARD    landed_ok=0.97   mean touchdown speed 3.00   soft landings 0.000
```

It reports success on 97% of episodes while slamming into the pad at near
free-fall speed. That is the original bug reproduced in isolation, and it says
plainly that **fixing the task was necessary and nowhere near sufficient.** The
lander only brakes because the reward asks it to.

### Second: the speed term has to be distance-gated

Charged flat, the speed term paralyses the agent instead of teaching it:
timeouts go from 2% to 60%. So the pair is the unit — the speed term teaches
braking, the gating stops it becoming a reason to hover. Neither half works
alone: flat gives 0.22 (mostly timeouts), absent gives 0.01 (mostly `too_fast`).

### Things the ablation overturned

**The landing speed gate does not teach braking.** Removing it costs nothing
measurable (0.58 vs 0.60, p=0.94) because the shaping already does that job. It
stays, because it is what *defines* success — without it the agent reports 0.82
`landed_ok` while only 0.58 are actually soft — but it is bookkeeping, not a
training signal. A terminal penalty tells the agent it failed and gives it no
gradient for fixing that.

**Fuel and time are substitutes, and neither is individually resolvable.**
Removing either alone lands inside the noise (p=0.38, p=0.40). Removing both
together is the one clearly harmful edit, pushing timeouts to 0.49 (p=0.036).
Something has to price stalling; it does not matter much which. Fuel looked
essential in earlier rounds only because it was tested *without* approach
gating, where it was the only thing preventing a stall.

### Learning curves

Soft-landing rate every 500k steps, mean over seeds — a variant that is merely
slower looks identical to one that never learns if you only read the last
number:

```
FULL REWARD (control)       0.19  0.13  0.24  0.33  0.42  0.56
minus time penalty          0.00  0.23  0.12  0.32  0.47  0.59
minus landing speed gate    0.01  0.16  0.35  0.42  0.44  0.64
minus fuel penalty          0.23  0.28  0.21  0.21  0.36  0.44
minus fuel AND time         0.00  0.50  0.03  0.00  0.30  0.24
minus approach gating       0.03  0.03  0.10  0.03  0.23  0.24
minus speed term in phi     0.00  0.03  0.00  0.03  0.04  0.01
banded shaping instead      0.00  0.00  0.00  0.00  0.00  0.00
no shaping at all           0.00  0.00  0.00  0.00  0.00  0.00
THE ORIGINAL REWARD         0.00  0.00  0.00  0.00  0.00  0.00
```

The bottom three never start — flat at 0.00 across the whole 3M. They are not
slow, they are pointed somewhere else. The control is still climbing at 3M, so
nothing here is converged and longer runs might reorder the middle.

---

## On "100%", and on variance

An earlier version of this write-up claimed 100% soft landings, "reproducible on
a second seed". That was two lucky draws, and it is worth being precise about
what is and is not true.

**The shipped checkpoint is genuinely that good.** `pretrained/ppo_lander_3M.zip`
scores 399/400 on fresh unseen seeds at a mean touchdown speed of 0.255, uses
thrust on 57% of steps, and degrades gracefully rather than falling over: 96.7%
at a 500px cap and **73.0% on the original uncapped spawn range** it was never
trained on.

**The recipe that produced it is not a 100% recipe.** Trained six times:

```
seed 0: 0.747   seed 1: 0.863   seed 2: 0.233
seed 3: 0.420   seed 4: 0.703   seed 5: 0.607

mean 0.596   sd 0.211   range 0.23-0.86
```

The shipped model is the lucky tail of that distribution. Expect a good run to
take a few attempts.

Two things were ruled out before blaming variance:

- **It is not the mid-training eval callback.** Re-running seeds 0–3 with the
  callback removed reproduced 0.747 / 0.863 / 0.233 / 0.420 — identical to three
  decimals.
- **Training is deterministic per seed on a given machine.** But the same seed
  on different hardware lands elsewhere in the range, because CPU float
  reduction order differs. The original 100% runs were on a different box.

This is the dominant effect in the whole project, and it is why only the
touchdown-speed split above is beyond argument. The ordering among the middle
rows should be read as a hint, not a result. Resolving fuel vs time vs gate
properly would need roughly 20 seeds each.

---

## Other bugs found along the way

Not reward design, but all real and all fixed:

- **A sign error made failure profitable.** An earlier commit had
  `reward -= -200` in every failure branch, so crashing, flying too high and
  going out of bounds each paid **+200**. Every way of failing was a bonus.
- **A γ residual paid the agent to exist.** `GAMMA * phi(s') - phi(s)` with
  `GAMMA=0.99` over a negative Φ leaves `-0.01·Φ` per step, which is *positive*
  and grows with distance from the pad: **+5 per step at 500px**, ~5000 across
  an episode, against a landing bonus of +500. Measured against fixed policies,
  hovering scored 6267 while a policy that actually landed 71 times in 150
  scored 3965. Value targets in the thousands with that spread is the value-loss
  explosion, in one line. It is also mis-specified: potential-based shaping is
  policy-invariant only when the shaping γ equals the agent's γ (0.99 vs 0.999
  here) and Φ(terminal) is 0 (it wasn't). The shaping is now a plain
  `phi(s') - phi(s)`, which has no residual to collect.
- **Truncation never set `TimeLimit.truncated`**, so SB3 scored every timeout as
  a real terminal worth zero regardless of what the lander had achieved.
- **Observations were raw pixels** — position on a ±1100 scale next to velocity
  on ±2.5, so velocity was ~500× smaller than the features beside it and an MLP
  could barely see it. One slot fed the network a constant (`target.y` is always
  `GROUND_Y`). Now scaled to roughly [-1, 1], with the dead slot carrying the
  lander's own height.
- **`outcomes/timeout` was missing** from the callback's status list, so
  timeouts were swept into `outcomes/other` — hiding the only failure mode a
  stalling policy produces.
- **`TIMESTEPS` was 10,000,000 per checkpoint**, which was also the only point
  at which ESC was polled, so the first save and the first chance to quit were
  both hours away.
- **The results screen counted three outcomes out of six**, so the rows never
  summed to the episode count.
- **`training_mode` recursed into `main_menu`** instead of returning to it,
  piling on a stack frame per visit.
- **The thrust sprite only drew for action 2**, never for up-left or up-right.
- **`matplotlib` was imported directly** but present only transitively via
  stable-baselines3.

---

## What I would do next

1. **Settle fuel vs time vs gate**, or stop carrying terms that cannot be shown
   to help. ~20 seeds each.
2. **Test whether `approach_scale` is still needed** with sanely weighted terms.
   If a 20-frame payback removes the wall, the flat speed term may work just as
   well and the code gets smaller. This was never tested — the jump from
   "found the wall" to "gate the speed term" skipped "weight it correctly first".
3. **Reduce the variance** before tuning anything else. At sd 0.21 most
   comparisons are unmeasurable. More parallel envs, a larger rollout, or a
   learning-rate schedule would all be worth trying.
4. **Train past 3M.** The control is still climbing where the runs stop.
5. **Revisit `SPEED_SCALE = 3.0`**, which was set from the 0.005 physics and is
   stale now that `ACCELERATION` is 0.01 (free-fall touchdown is 3.44, observed
   peaks 4.23). It is baked into the shipped policy's input scaling, so changing
   it requires a retrain.

---

## Reproducing any of this

```bash
python tools/experiment.py --list                  # every configuration
python tools/experiment.py final --timesteps 3000000

# the ablation, one term at a time, several seeds each
python tools/experiment.py abl_full --seed 0 --timesteps 3000000 --eval-every 500000
python tools/ablation_report.py results/ablation

python -m unittest discover -s tests               # 24 regression tests
```

| file | what it holds |
|---|---|
| `game_core/game_logic.py` | `RewardConfig` — every tunable term, with the reasoning next to it |
| `tools/experiment.py` | the harness; each `EXPERIMENTS` entry is one hypothesis |
| `tools/ablation_report.py` | the tables above, with permutation tests |
| `results/*.json` | the chronological runs |
| `results/ablation/*.json` | all 40 ablation runs |
| `tests/test_lander.py` | regression tests, mostly for bugs that were once silent |

`RewardConfig.shaping` takes `"potential"`, `"banded"` or `"none"`, so the
historical shaping can be trained against rather than argued about. To get the
old task back: `SPAWN_MAX_GAP = None`, `ACCELERATION = 0.005`,
`fuel_penalty = 0.0`, `landing_speed_limit = math.inf`, `shaping = "banded"`.

A note on the yardstick itself: every reward in this document agrees it is doing
well by its own accounting. The original one reports 97% landings while falling
out of the sky. Scoring against something the reward cannot influence is most of
what made any of this legible.
