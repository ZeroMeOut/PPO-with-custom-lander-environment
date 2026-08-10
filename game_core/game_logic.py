import sys
import math
import random
import pygame
import numpy as np
from numpy import ndarray
from dataclasses import dataclass, replace
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

## How far the pad may be placed from the lander's spawn, horizontally.
##
## Drawing both ends independently across the full 1180px range asks for
## something the lander cannot do. The booster accelerates at 0.005 px/frame^2,
## so an accelerate-then-decelerate crossing covers only 0.00125 * T^2 px: 118px
## over the 307 frame free fall, and 312px even over 500 frames. Against a mean
## gap of 391px and a p90 of 804px, 81% of episodes began beyond reach unless the
## lander hovered to buy time, and a trained agent scored 12% inside 200px and
## 0.0% past it. That is not a policy failing to learn, it is a task that mostly
## cannot be flown.
##
## Capping the gap keeps both ends spawning from the same range -- so the
## left-bias this range was widened to fix stays fixed -- while keeping the pad
## somewhere the lander can actually get to. None restores the old behaviour.
SPAWN_MAX_GAP: Optional[int] = 250

## Acceleration applied per frame, by gravity and by the booster alike. They are
## equal, so the lander can decelerate exactly as fast as it falls.
##
## The game shipped with 0.005 and that is the one number here that changes how
## it feels to play, so it is worth being explicit about why it moved. Over 3M
## steps on the capped geometry, 0.005 lands 38.7% of episodes while 0.01 lands
## 100% of them, twice, on two seeds. At 0.005 a free fall lasts 307 frames and
## covers 118px of ground, which is less than the lander is asked to cross; at
## 0.01 it covers 209px and the task closes. Put it back to 0.005 if you prefer
## the original handling -- 38.7% is still a lander that flies, and the reward
## fixes are what got it there.
ACCELERATION: float = 0.01

## get_font re-reads the .ttf from disk on every call, so build the HUD font once.
HUD_FONT = get_font(15)

## Scales that turn raw pixel quantities into roughly unit-sized numbers, used by
## both the potential and the observation. GROUND_Y is where the pad sits.
GROUND_Y: int = 513
DIST_SCALE: float = 1400.0   ## ~hypot(1280, 600), the longest possible distance
SPEED_SCALE: float = 3.0     ## free fall reaches ~2.5 px/frame before touchdown


@dataclass(frozen=True)
class RewardConfig:
    """Every tunable term of the reward, in one place so runs can be compared.

    The shaping is potential based: F = phi(s') - phi(s). Written as a plain
    difference rather than the textbook gamma * phi(s') - phi(s) on purpose.
    With gamma < 1 the discounted form leaves a residual (gamma - 1) * phi per
    step, and since phi is negative that residual is *positive* -- a small
    payment for merely staying alive, which is exactly the incentive that makes
    an agent hover instead of land. The plain difference telescopes exactly to
    phi(end) - phi(start), so the shaping total depends only on where the
    episode started and finished and cannot be farmed by any trajectory.
    """
    ## Potential weights. w_speed is what makes braking pay: without it nothing
    ## in the reward distinguishes a controlled descent from a free fall.
    w_dist: float = 100.0
    w_speed: float = 50.0

    ## How much speed the lander is allowed before w_speed starts charging for
    ## it: v_ref = clamp(distance / approach_scale, approach_v_min, approach_v_max).
    ##
    ## Charging for speed everywhere, which is what approach_scale = 0 does,
    ## builds a wall around every hovering state, and this rewrite walked
    ## straight into it. Distance and speed enter the potential independently,
    ## but speed is the derivative of distance, so closing the gap requires
    ## carrying the thing being penalised. At w_dist=100 over DIST_SCALE=1400
    ## against w_speed=50 over SPEED_SCALE=3, a unit of speed costs 16.67 while
    ## a pixel earns 0.0714: ~233px had to be closed before moving was worth it,
    ## and nearer than that no route to the pad paid for itself. Every config
    ## tried against that potential converged on hovering until the clock ran
    ## out, which is exactly the behaviour it rewards. (The banded shaping this
    ## replaced had the opposite flaw -- no speed term at all.)
    ##
    ## Making the allowance proportional to distance says the sensible thing
    ## instead: far away, travel fast; close in, arrive slowly. Approaching the
    ## pad at a reasonable speed is then free, and only the excess is charged.
    approach_scale: float = 200.0
    approach_v_min: float = 0.2
    approach_v_max: float = 2.5

    ## Small per-step cost. Bounds the value of stalling near the pad, so
    ## running out the clock cannot beat committing to a landing.
    time_penalty: float = 0.05

    ## Cost of firing the booster, charged per frame of thrust. This is the term
    ## that actually stops the agent hovering, and it took a sweep to find that
    ## out. A plain time penalty prices *all* delay equally, so raising it enough
    ## to forbid a stall also rushes the descent and the lander crashes. Fuel
    ## prices only the delay that costs thrust: holding altitude means burning
    ## every frame, while falling and braking late is nearly free. The real
    ## Lunar Lander charges the same way, and its absence here is why hovering
    ## was the strongest policy the agent could find.
    ##
    ## 0.05 came out of a sweep against fixed policies. The booster is weak
    ## (0.005 per frame against the same gravity), so a real descent spends
    ## hundreds of frames thrusting; past about 0.2 the bill for braking grows
    ## until falling like a stone scores as well as flying, which is the
    ## opposite of the point.
    fuel_penalty: float = 0.05

    ## Terminal payouts. landing_bonus is deliberately much larger than the
    ## shaping an episode can bank (~60), so landing dominates every
    ## alternative rather than merely edging it out.
    landing_bonus: float = 200.0
    too_fast_penalty: float = 50.0     ## on the pad but coming in hot
    crash_penalty: float = 100.0       ## hit the ground off the pad
    high_penalty: float = 100.0
    oob_penalty: float = 100.0
    timeout_penalty: float = 0.0       ## see LanderEnvironment.step

    ## Impact speed at or below which touching the pad counts as a landing.
    landing_speed_limit: float = 1.0


REWARD: RewardConfig = RewardConfig()


def set_reward_config(config: RewardConfig) -> None:
    global REWARD
    REWARD = config

def set_render_enabled(enabled: bool) -> None:
    global RENDER_ENABLED
    RENDER_ENABLED = enabled

def set_random_target(random_target: bool) -> None:
    global RANDOM_TARGET
    RANDOM_TARGET = random_target

def set_spawn_max_gap(max_gap: Optional[int]) -> None:
    global SPAWN_MAX_GAP
    SPAWN_MAX_GAP = max_gap

def set_acceleration(acceleration: float) -> None:
    global ACCELERATION
    ACCELERATION = acceleration

def speed(gs) -> float:
    """Current speed of the lander. x/y_acceleration are really velocities: move()
    adds them straight to the position, so they are px per frame."""
    return math.hypot(gs.player.x_acceleration, gs.player.y_acceleration)


def distance(gs) -> float:
    return math.hypot(gs.player.x - gs.target.x, gs.player.y - gs.target.y)


def reference_speed(d: float) -> float:
    """How fast the lander may travel at distance d before speed starts costing.

    Grows with distance, so the potential asks for a fast cruise and a slow
    arrival rather than for slowness everywhere. See RewardConfig.approach_scale.
    """
    if REWARD.approach_scale <= 0:
        return 0.0
    return min(max(d / REWARD.approach_scale, REWARD.approach_v_min),
               REWARD.approach_v_max)


def phi(gs) -> float:
    """Potential of a state: higher is better, zero is sitting still on the pad.

    A true function of the state, unlike the banded version this replaces, which
    mutated gs and paid a flat +/-10 per band crossing. Bands were 54-125px wide,
    so the whole final approach sat inside band 0 with no gradient at all -- the
    agent got no signal precisely where precision matters.
    """
    d: float = distance(gs)
    excess: float = max(0.0, speed(gs) - reference_speed(d))
    return -(REWARD.w_dist * d / DIST_SCALE
             + REWARD.w_speed * excess / SPEED_SCALE)


def calculate_reward_and_done(gs, drawing: bool = True):
        ## Potential-based shaping, plus the cost of living and of burning fuel.
        ## See RewardConfig for why both exist.
        current_phi: float = phi(gs)
        reward: float = current_phi - gs.prev_phi - REWARD.time_penalty
        if gs.is_thrusting:
            reward -= REWARD.fuel_penalty
        gs.prev_phi = current_phi

        done: bool = False
        info: Dict[str, Any] = {}

        ## Terminal states only flag the episode as over. Resetting here would
        ## overwrite the state before run_game_frame builds the observation, so
        ## the caller was handed the next episode's spawn point as the terminal
        ## observation. Respawning belongs to reset().
        if gs.player.y > GROUND_Y:
            impact: float = speed(gs)
            if gs.player.collided_with(gs.target):
                ## Reaching the pad is not the same as landing on it. Without
                ## this speed gate a full-speed free fall onto the pad scored a
                ## clean success, so nothing in the reward ever asked the agent
                ## to brake -- which is exactly what it learned not to do.
                done = True
                if impact <= REWARD.landing_speed_limit:
                    reward += REWARD.landing_bonus
                    info["status"] = "landed_ok"
                else:
                    if drawing:
                        display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
                    ## Penalised less than a miss: arriving at the right place
                    ## too fast is a better failure than arriving nowhere near,
                    ## and grading it that way gives the agent a route in.
                    reward -= REWARD.too_fast_penalty
                    info["status"] = "too_fast"
            else:
                if drawing:
                    display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
                done = True
                reward -= REWARD.crash_penalty
                info["status"] = "crashed"
            info["impact_speed"] = impact

        elif gs.player.y < -50:
            if drawing:
                display_image(EXPLOSION_IMAGE, gs.player.x - 17, gs.player.y - 18)
            done = True
            reward -= REWARD.high_penalty
            info["status"] = "flown_too_high"

        ## elif, not if: now that this reads the real position rather than a
        ## post-reset one, a frame that reaches the ground must keep its
        ## outcome instead of being relabelled out of bounds.
        elif gs.player.x < 0 or gs.player.x > 1280:
            done = True
            reward -= REWARD.oob_penalty
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
        ## Whether the booster fired on the frame being scored, set by
        ## run_game_frame so the reward can charge fuel for it.
        self.is_thrusting: bool = False
        self.prev_phi: float = phi(self)

    def seed(self, seed: int) -> None:
        """Reseed the game RNG so the next reset is reproducible."""
        self.rng.seed(seed)

    def _target_x(self) -> int:
        ## Read at call time so menu choices apply to the next reset.
        if not RANDOM_TARGET:
            return STATIC_TARGET_X
        low, high = SPAWN_X_MIN, SPAWN_X_MAX
        if SPAWN_MAX_GAP is not None:
            ## Clamped to the spawn range, so the pad stays on screen. Clamping
            ## can make the window one sided near an edge, which is fine: it is
            ## still centred on the lander wherever there is room to be.
            low = max(low, int(self.player.x) - SPAWN_MAX_GAP)
            high = min(high, int(self.player.x) + SPAWN_MAX_GAP)
        return self.rng.randint(low, high)

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
        self.is_thrusting = False
        ## Must be restored like the positions above. Left stale, the first step
        ## of an episode is paid against the previous episode's final state.
        self.prev_phi = phi(self)

    def get_observation(self) -> ndarray:
        """Build an observation from the current state without advancing the game.

        Scaled to roughly [-1, 1]. Raw pixels put position on a +/-1100 scale and
        velocity on a +/-2.5 one, so velocity was ~500x smaller than the features
        beside it and an MLP could barely see it -- a poor position to learn
        braking from even once the reward asks for it.

        The last slot used to be target.y, which is always GROUND_Y and so fed
        the network a constant. It carries the lander's own height instead, and
        player.x replaces target.x so the agent can see the walls it is
        terminated for touching.
        """
        return np.array([
            (self.player.x - self.target.x) / 640.0,
            (self.player.y - self.target.y) / 600.0,
            self.player.x_acceleration / SPEED_SCALE,
            self.player.y_acceleration / SPEED_SCALE,
            self.player.x / 640.0 - 1.0,
            self.player.y / 600.0,
        ])

def run_game_frame(
    gs: GameState,
    mode: str,
    action: Optional[int] = None
) -> Optional[Tuple[ndarray, float, bool, Dict[str, Any]]]:
    player_x_acceleration: float = 0.0
    player_y_acceleration: float = ACCELERATION  ## gravity, unless thrust overrides it

    ## Drawing is skipped entirely when nothing will be shown. It is not just
    ## wasted work: with several envs in one process they all share SCREEN, so
    ## they would be scribbling over each other's frames for no reason.
    drawing: bool = mode != "training" or RENDER_ENABLED

    if drawing:
        SCREEN.blit(GAME_BG, (0, 0))

        ## Debug readout. Must stay inside `if drawing`: built unconditionally it
        ## rendered text on every training step and cost ~150x of the throughput.
        LANDER_ACCELERATION: TextObject = TextObject(
            text_input=f"Acceleration: {gs.player.x_acceleration:.3f}, {gs.player.y_acceleration:.3f}",
            font=HUD_FONT,
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
                    player_x_acceleration = -ACCELERATION
                    gs.is_left_pressed = True
                if event.key == pygame.K_RIGHT:
                    player_x_acceleration = ACCELERATION
                    gs.is_right_pressed = True
                if event.key == pygame.K_UP:
                    player_y_acceleration = -ACCELERATION
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
            player_x_acceleration = -ACCELERATION
        elif gs.is_right_pressed:
            player_x_acceleration = ACCELERATION
        if gs.is_up_pressed:
            player_y_acceleration = -ACCELERATION
        else:
            player_y_acceleration = ACCELERATION

    elif mode in ["training", "test"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if action == 0:  # left
            player_x_acceleration = -ACCELERATION
            player_y_acceleration = ACCELERATION
        elif action == 1:  # right
            player_x_acceleration = ACCELERATION
            player_y_acceleration = ACCELERATION
        elif action == 2:  # up
            player_x_acceleration = 0.0
            player_y_acceleration = -ACCELERATION
        elif action == 3:  # up left
            player_x_acceleration = -ACCELERATION
            player_y_acceleration = -ACCELERATION
        elif action == 4:  # up right
            player_x_acceleration = ACCELERATION
            player_y_acceleration = -ACCELERATION
        elif action == 5:  # no action
            player_x_acceleration = 0.0
            player_y_acceleration = ACCELERATION

    ## Any upward acceleration means the booster fired, whichever mode and
    ## whichever of the three thrusting actions produced it. Derived from the
    ## acceleration rather than compared against action == 2, which used to miss
    ## the up-left and up-right actions and so drew them without a flame.
    gs.is_thrusting = player_y_acceleration < 0

    gs.player.x_acceleration += player_x_acceleration
    gs.player.y_acceleration += player_y_acceleration
    gs.player.move()
    gs.player.rect.topleft = (int(gs.player.x), int(gs.player.y))

    if drawing:
        gs.target.display(SCREEN)
        if gs.is_thrusting:
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