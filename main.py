from architectures.pinn import Pinn
from architectures.xtfc_unfreeze import XTFC_Unfreeze
from architectures.xtfc import XTFC
from models.hparams import Hyperparams
from problems.inverted_pendulum import inverted_pendulum
from problems.damped_inverted_pendulum import damped_inverted_pendulum
from models.simulator import Simulator
from visualizers.pendulum import PendulumVisualizer
import torch
import log
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd

def generate_trajectory(x0: torch.Tensor, t_span: np.ndarray, time_step: float, min_delta: float = None, patience: int = None, zero_control: bool = False) -> torch.Tensor:
        trajectory = [x0]
        u = []

        x_current = x0
        n_steps = int(t_span / time_step)

        counter = patience
        converged = False

        for step in tqdm(range(n_steps), desc="Generating trajectory", unit="step", ncols=80):
            x_current.requires_grad_(True)
            _, _, _, grad_v = model.get_outputs(x_current)
            f_x = model.problem.f_x(x_current)
            g_x = model.problem.g_x(x_current)

            if not zero_control:
                u_star = model.problem.control_input(x_current, grad_v)
            else:
                u_star = torch.tensor([[0.0]], device=model.device)

            x_dot = f_x + g_x * u_star
            x_next = x_current + time_step * x_dot

            if model.hparams.hyper_params.problem.lower() == "inverted-pendulum":
                x_next[:, 0] = (x_next[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi

            diff = torch.norm(x_next - model.problem.eq_point).item()
            if min_delta is not None and patience is not None:
                if diff < min_delta:
                    counter -= 1
                    if counter == 0:
                        model.logger.info(f"Early stopping at step {step} with delta {diff:.6f}")
                        converged = True
                        break
                else:
                    counter = patience

            trajectory.append(x_next)
            u.append(u_star)
            x_current = x_next

        return torch.cat(trajectory, dim=0), torch.cat(u, dim=0), converged
    

if __name__ == "__main__":
    Hyperparams_obj = Hyperparams.from_yaml("yamls/unfreeze_ip.yaml")
    logger = log.get_logger("main")
    logger.setLevel(logging.INFO if Hyperparams_obj.hyper_params.debug == False else logging.DEBUG)
    Hyperparams_obj.logger = logger

    problem = damped_inverted_pendulum(Hyperparams_obj)
    model = XTFC_Unfreeze(problem)
    model.to(device=model.device)

    model.analytical_pretraining()
    model.train_model()

    simulator = Simulator(model)
    x0 = torch.tensor([[0.1, 1.0]], dtype=torch.float32, device=model.device)

    

    # Generate trajectory and controls
    traj, controls, converged = generate_trajectory(
        x0,
        10.0,
        time_step=0.01,
    )

    # Save trajectory and controls to CSV with timestep, state1, state2, u
    import pandas as pd
    traj_np = traj.detach().cpu().numpy()
    controls_np = controls.detach().cpu().numpy()
    # Prepend NaN to controls to align with trajectory
    controls_np = np.vstack([np.full((1, controls_np.shape[1]), np.nan), controls_np])
    # Create timestep column
    dt = 0.01
    timesteps = np.arange(traj_np.shape[0]) * dt
    # Concatenate all columns
    data = np.hstack([timesteps.reshape(-1, 1), traj_np, controls_np])
    columns = ["timestep", "state1", "state2", "u"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("trajectory_output.csv", index=False)
    print("Trajectory and controls saved to trajectory_output.csv with columns: timestep, state1, state2, u")



    # simulator.test_model(
    #     n_points=10,
    #     t_span=10.0,
    #     time_step=0.01,
    #     min_delta=1e-2,
    #     patience=50,
    #     random=True,
    #     ranges = [[-np.pi, np.pi], [-5.0, 5.0]],
    #     plot=True
    # )

    # visualizer = PendulumVisualizer(model, problem, time_step=0.01, initial_state=x0)
    # visualizer.run()