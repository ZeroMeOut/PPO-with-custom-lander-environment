import matplotlib
## Training is headless, and this module is only imported to train, so pick a
## non-interactive backend before pyplot is loaded.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

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

    def __init__(self, bar_chart: bool = True, figure_freq: int = 1, verbose: int = 0):
        """
        bar_chart:   also log a labelled bar chart under actions/distribution.
                     Tensorboard has no bar chart dashboard for scalars, so this
                     goes to the IMAGES tab with a slider over training steps.
        figure_freq: log that chart every Nth rollout. Each figure is a PNG in
                     the event file, so raise this if the logs get too big.
        """
        super().__init__(verbose)
        self.counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
        self.bar_chart = bar_chart
        self.figure_freq = max(1, figure_freq)
        self._rollouts = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            flat = np.asarray(actions).astype(np.int64).ravel()
            self.counts += np.bincount(flat, minlength=len(ACTION_NAMES))
        return True

    def _make_bar_chart(self, freqs: np.ndarray) -> plt.Figure:
        """Bar chart of the action mix, booster actions highlighted."""
        fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
        colours = ["tab:orange" if i in THRUST_ACTIONS else "tab:blue"
                   for i in range(len(ACTION_NAMES))]
        bars = ax.bar(ACTION_NAMES, freqs, color=colours)
        for bar, freq in zip(bars, freqs):
            ax.text(bar.get_x() + bar.get_width() / 2, freq + 0.02, f"{freq:.2f}",
                    ha="center", va="bottom", fontsize=8)

        ## A uniform policy sits on this line, so anything below it on the
        ## orange bars means the agent has actively learned not to thrust.
        ax.axhline(1 / len(ACTION_NAMES), ls="--", lw=1, color="grey")
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of steps")
        ax.set_title(f"action mix @ {self.num_timesteps:,} timesteps"
                     f"   (thrust {freqs[list(THRUST_ACTIONS)].sum():.2f}, orange)")
        ax.spines[["top", "right"]].set_visible(False)
        return fig

    def _on_rollout_end(self) -> None:
        total = int(self.counts.sum())
        if total:
            freqs = self.counts / total
            for name, freq in zip(ACTION_NAMES, freqs):
                self.logger.record(f"actions/{name}", float(freq))
            self.logger.record("actions/thrust_any", float(freqs[list(THRUST_ACTIONS)].sum()))

            if self.bar_chart and self._rollouts % self.figure_freq == 0:
                ## A Figure cannot be written to stdout/csv/json, only tensorboard.
                self.logger.record("actions/distribution",
                                   Figure(self._make_bar_chart(freqs), close=True),
                                   exclude=("stdout", "log", "json", "csv"))
        self._rollouts += 1
        ## Each point describes one rollout rather than all of training so far.
        self.counts[:] = 0
