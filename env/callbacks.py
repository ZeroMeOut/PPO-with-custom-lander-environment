import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

## Matches the action space in LanderEnvironment / run_game_frame:
## 0: left, 1: right, 2: up, 3: up left, 4: up right, 5: do nothing
ACTION_NAMES = ["left", "right", "up", "up_left", "up_right", "nothing"]

## Every action that fires the booster upward. Logged as one series so you can
## see at a glance whether the policy is braking at all.
THRUST_ACTIONS = (2, 3, 4)


class ActionFrequencyCallback(BaseCallback):
    """Log how often the policy picks each action, as a fraction per rollout.

    Written to tensorboard under actions/<name>, plus actions/thrust_any for the
    combined share of the three booster actions. The fractions over the named
    actions sum to 1.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            flat = np.asarray(actions).astype(np.int64).ravel()
            self.counts += np.bincount(flat, minlength=len(ACTION_NAMES))
        return True

    def _on_rollout_end(self) -> None:
        total = int(self.counts.sum())
        if total:
            freqs = self.counts / total
            for name, freq in zip(ACTION_NAMES, freqs):
                self.logger.record(f"actions/{name}", float(freq))
            self.logger.record("actions/thrust_any", float(freqs[list(THRUST_ACTIONS)].sum()))
        ## Each point describes one rollout rather than all of training so far.
        self.counts[:] = 0
