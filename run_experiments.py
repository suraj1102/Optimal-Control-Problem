from experiment import (
    MODEL_DIR,
    PLOT_DIR,
    generate_reward_grid,
    train_experiment,
    plot_training_curves,
)
import os
import numpy as np

from training.disturbances import DISTURB_FNS

BASE_ENV_KWARGS = dict(
    dt=0.02,
    max_steps=1000,
    action_low=-1.0,
    action_high=1.0,
    init_range=((-0.5, 0.5), (-0.5, 0.5)),
    damping_factor=0.1,
    gravity=10,
    length=0.8,
    mass=0.1,
    failure_termination=(np.pi, 10.0),
    success_termination=None,
)


def run_experiments(experiments):
    all_results: dict[str, dict] = {}

    for exp in experiments:
        result = train_experiment(exp)
        all_results[exp["run_name"]] = result

    plot_training_curves(
        all_results,
        save_path=os.path.join(PLOT_DIR, "training_curves.png"),
    )
    print("\n[done] all experiments complete.")
    print(f"  plots  → {PLOT_DIR}/")
    print(f"  models → {MODEL_DIR}/")


if __name__ == "__main__":
    rewards = generate_reward_grid(
        Q1_vals=[1.0, 3.0, 5.0],
        Q2_vals=[0.1, 0.5, 1.0],
        R_vals=[0.01],
        norms=[False, True],
        survival_thresholds=[3, 6, 12],
    )

    disturbances = {
        "no_disturbance": DISTURB_FNS["none"],
        "gaussian": DISTURB_FNS["gaussian"],
        "impulse": DISTURB_FNS["impulse"],
    }
