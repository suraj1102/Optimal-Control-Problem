import stable_baselines3 as sb3
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from invertedpendulum import InvertedPendulumEnv
from training.disturbances import DISTURB_FNS
from training.rewards import make_reward_quadratic

MODEL_PATH = "../outputs/models/DDPG_Q1_2.0_Q2_0.1_R_0.5_no_disturbance"

dt = 0.01
dtheta = 0.015
dthetadot = 0.015

theta_min, theta_max = -np.pi, np.pi
thetadot_min, thetadot_max = -2.0, 2.0


BASE_ENV_KWARGS = dict(
    dt=0.01,
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


def round_to_nearest(x, base=0.015):
    return float(np.round(x / base) * base)


def main():
    model = sb3.DDPG.load(MODEL_PATH)

    env = InvertedPendulumEnv(
        reward_fn=make_reward_quadratic(2, 0.1, 0.5, normalise=False),
        disturb_fn=DISTURB_FNS["none"],
        **BASE_ENV_KWARGS,
    )

    theta_range = np.arange(theta_min, theta_max + 0.5 * dtheta, dtheta)
    thetadot_range = np.arange(thetadot_min, thetadot_max + 0.5 * dthetadot, dthetadot)

    data = []
    for theta in theta_range:
        for thetadot in thetadot_range:
            state = np.array([theta, thetadot], dtype=np.float32)
            obs = state.copy()
            # Model expects batch dimension
            action, _ = model.predict(obs, deterministic=True)
            # Step dynamics manually to get next_theta
            next_state = env._dynamics(state, action)
            next_theta = next_state[0]
            data.append(
                {
                    "theta": round(round_to_nearest(theta, dtheta), 3),
                    "thetadot": round(round_to_nearest(thetadot, dthetadot), 3),
                    "next_theta": round(round_to_nearest(next_theta, dtheta), 6),
                }
            )

    df = pd.DataFrame(data)
    df.to_csv("policy_table.csv", index=False)
    print("Saved table to policy_table.csv")


if __name__ == "__main__":
    main()
