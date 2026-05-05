import stable_baselines3 as sb3
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from invertedpendulum import InvertedPendulumEnv
from training.disturbances import DISTURB_FNS
from training.rewards import make_reward_quadratic

MODEL_PATH = "../outputs/models/DDPG_Q1_2.0_Q2_0.1_R_0.5_no_disturbance"
OUTPUT_DIR = "outputs"
FILENAME = "ddpg_rollout"

dt = 0.01
tf = 10 #s 
numPoints = int(tf / dt) + 1

BASE_ENV_KWARGS = dict(
    dt=dt,
    max_steps=numPoints,
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

theta = 0.1
thetadot = 1

def main():
    model = sb3.DDPG.load(MODEL_PATH)

    env = InvertedPendulumEnv(
        reward_fn=make_reward_quadratic(2, 0.1, 0.5, normalise=False),
        disturb_fn=DISTURB_FNS["none"],
        **BASE_ENV_KWARGS,
    )
    
    
    # Rollout storage
    thetas = []
    thetadots = []
    controls = []
    rewards = []

    state = np.array([theta, thetadot], dtype=np.float32)
    for _ in range(numPoints):
        obs = state.copy()
        action, _ = model.predict(obs, deterministic=True)
        next_state = env._dynamics(state, action)
        reward = env._reward(state, action)

        thetas.append(state[0])
        thetadots.append(state[1])
        controls.append(action[0] if isinstance(action, np.ndarray) else action)
        rewards.append(reward)

        state = next_state

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame({
        'theta': thetas,
        'thetadot': thetadots,
        'control': controls,
        'reward': rewards
    })
    df.to_csv(f'{OUTPUT_DIR}/{FILENAME}.csv', index=False)

    # Plot
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

    axs[0].plot(thetas)
    axs[0].set_ylabel('theta')
    axs[0].grid(True)

    axs[1].plot(thetadots)
    axs[1].set_ylabel('thetadot')
    axs[1].grid(True)

    axs[2].plot(controls)
    axs[2].set_ylabel('control')
    axs[2].grid(True)

    axs[3].plot(rewards)
    axs[3].set_ylabel('reward')
    axs[3].set_xlabel('Timestep')
    axs[3].grid(True)

    fig.suptitle(f'{FILENAME} Results')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{FILENAME}.png")
    plt.show()
        

if __name__ == "__main__":
    main()
