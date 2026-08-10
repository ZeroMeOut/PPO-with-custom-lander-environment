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

## Every info["status"] calculate_reward_and_done and the env can produce.
## Listed explicitly so the tensorboard tags exist from the first rollout
## rather than appearing only once an outcome first happens.
STATUS_NAMES = [
    "landed_ok",
    "crashed",
    "flown_too_high",
    "out_of_horizontal_bounds",
    "too_fast",
    ## Timeouts are set by the env rather than calculate_reward_and_done, and
    ## were missing here, so they were silently swept into outcomes/other --
    ## hiding the one failure mode a stalling policy actually produces.
    "timeout",
]

## Short forms for the chart only. The tensorboard tags keep the full names;
## these just stop the x axis turning into a wall of text.
STATUS_LABELS = {
    "landed_ok": "landed",
    "crashed": "crashed",
    "flown_too_high": "too high",
    "out_of_horizontal_bounds": "out of bounds",
    "timeout": "timeout",
    "other": "other",
}

## The one outcome that counts as success, so it can be coloured apart.
SUCCESS_STATUS = "landed_ok"


def make_bar_chart(labels, freqs, colours, title, baseline=None,
                   ylabel="fraction") -> plt.Figure:
    """Labelled bar chart of a distribution, for logging to tensorboard.

    baseline draws a dashed reference line, for when there is a meaningful
    "no preference" level to compare against. Outcomes have no such level, so
    they pass None.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    bars = ax.bar(labels, freqs, color=colours)
    for bar, freq in zip(bars, freqs):
        ## A label above a near-full bar would land outside the axes and collide
        ## with the title, which is exactly the case worth reading, so put the
        ## label inside the bar instead once it gets tall.
        inside = freq > 0.92
        ax.text(bar.get_x() + bar.get_width() / 2,
                freq - 0.03 if inside else freq + 0.02, f"{freq:.2f}",
                ha="center", va="top" if inside else "bottom", fontsize=8,
                color="white" if inside else "black")

    if baseline is not None:
        ax.axhline(baseline, ls="--", lw=1, color="grey")

    ## Fixed rather than autoscaled, so frames stay comparable while scrubbing
    ## the step slider in the IMAGES tab.
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


class ActionFrequencyCallback(BaseCallback):
    """Log how often the policy picks each action, as a fraction per rollout.

    Written to tensorboard as a bar chart under actions/distribution, with the
    booster actions highlighted so it is obvious whether the policy is braking.
    """

    def __init__(self, bar_chart: bool = True, figure_freq: int = 1, verbose: int = 0):
        """
        bar_chart:   log a labelled bar chart under actions/distribution.
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

    def _on_rollout_end(self) -> None:
        total = int(self.counts.sum())
        if total:
            freqs = self.counts / total

            if self.bar_chart and self._rollouts % self.figure_freq == 0:
                colours = ["tab:orange" if i in THRUST_ACTIONS else "tab:blue"
                           for i in range(len(ACTION_NAMES))]
                thrust = freqs[list(THRUST_ACTIONS)].sum()
                fig = make_bar_chart(
                    ACTION_NAMES, freqs, colours,
                    ## A uniform policy sits on the baseline, so orange bars
                    ## below it mean the agent actively learned not to thrust.
                    title=f"action mix @ {self.num_timesteps:,} timesteps"
                          f"   (thrust {thrust:.2f}, orange)",
                    baseline=1 / len(ACTION_NAMES),
                    ylabel="fraction of steps",
                )
                ## A Figure cannot be written to stdout/csv/json, only tensorboard.
                self.logger.record("actions/distribution", Figure(fig, close=True),
                                   exclude=("stdout", "log", "json", "csv"))
        self._rollouts += 1
        ## Each point describes one rollout rather than all of training so far.
        self.counts[:] = 0


class EpisodeOutcomeCallback(BaseCallback):
    """Log how episodes ended, as fractions per rollout.

    Written to tensorboard under outcomes/<status>, and as a bar chart under
    outcomes/distribution. ep_rew_mean is dominated by distance shaping, so it
    tracks "got close" rather than "landed"; this is the series that answers
    whether the agent is actually succeeding.
    """

    def __init__(self, bar_chart: bool = True, figure_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.counts: dict[str, int] = {}
        self.bar_chart = bar_chart
        self.figure_freq = max(1, figure_freq)
        self._rollouts = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            ## VecEnv auto-resets on done but keeps the env's own info dict, so
            ## the status set by calculate_reward_and_done survives to here.
            if done:
                status = info.get("status", "unknown")
                self.counts[status] = self.counts.get(status, 0) + 1
        return True

    def _on_rollout_end(self) -> None:
        total = sum(self.counts.values())
        ## Skip rather than log zeros when no episode finished this rollout,
        ## so a gap in the series is not mistaken for a 0% success rate.
        if total:
            for name in STATUS_NAMES:
                self.logger.record(f"outcomes/{name}", self.counts.get(name, 0) / total)
            unexpected = sum(v for k, v in self.counts.items() if k not in STATUS_NAMES)
            if unexpected:
                self.logger.record("outcomes/other", unexpected / total)

            # if self.bar_chart and self._rollouts % self.figure_freq == 0:
            #     ## Same bars as the scalars above, including the "other" bucket
            #     ## only when something landed in it, so the two always agree.
            #     names = list(STATUS_NAMES) + (["other"] if unexpected else [])
            #     freqs = np.array([self.counts.get(n, 0) / total for n in STATUS_NAMES]
            #                      + ([unexpected / total] if unexpected else []))
            #     labels = [STATUS_LABELS.get(n, n) for n in names]
            #     colours = ["tab:green" if n == SUCCESS_STATUS
            #                else "tab:grey" if n == "other" else "tab:red"
            #                for n in names]
            #     landed = self.counts.get(SUCCESS_STATUS, 0) / total
            #     fig = make_bar_chart(
            #         labels, freqs, colours,
            #         title=f"episode outcomes @ {self.num_timesteps:,} timesteps"
            #               f"   (landed {landed:.2f} of {total} episodes)",
            #         ylabel="fraction of episodes",
            #     )
            #     self.logger.record("outcomes/distribution", Figure(fig, close=True),
            #                        exclude=("stdout", "log", "json", "csv"))
        self._rollouts += 1
        self.counts.clear()