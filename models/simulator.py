from models.valuefunctionmodel import ValueFunctionModel
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Simulator():
    def __init__(self, model):
        self.model = model

    
    def generate_trajectory(self, x0: torch.Tensor, t_span: np.ndarray, time_step: float, min_delta: float = None, patience: int = None, zero_control: bool = False) -> torch.Tensor:
        trajectory = [x0]
        u = []

        x_current = x0
        n_steps = int(t_span / time_step)

        counter = patience
        converged = False

        for step in tqdm(range(n_steps), desc="Generating trajectory", unit="step", ncols=80):
            x_current.requires_grad_(True)
            _, _, _, grad_v = self.model.get_outputs(x_current)
            f_x = self.model.problem.f_x(x_current)
            g_x = self.model.problem.g_x(x_current)

            if not zero_control:
                u_star = self.model.problem.control_input(x_current, grad_v)
            else:
                u_star = torch.tensor([[0.0]], device=self.model.device)

            x_dot = f_x + g_x * u_star
            x_next = x_current + time_step * x_dot

            if self.model.hparams.hyper_params.problem.lower() == "inverted-pendulum":
                x_next[:, 0] = (x_next[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi

            diff = torch.norm(x_next - self.model.problem.eq_point).item()
            if min_delta is not None and patience is not None:
                if diff < min_delta:
                    counter -= 1
                    if counter == 0:
                        self.model.logger.info(f"Early stopping at step {step} with delta {diff:.6f}")
                        converged = True
                        break
                else:
                    counter = patience

            trajectory.append(x_next)
            u.append(u_star)
            x_current = x_next

        return torch.cat(trajectory, dim=0), torch.cat(u, dim=0), converged
    

    def test_model(self, n_points, t_span, time_step, min_delta=None, patience=None, random=True, ranges=None, plot=False):
        success_count = 0

        ranges = self.model.hparams.problem_params.input_ranges if ranges is None else ranges

        if random:
            x0_samples = []
            for _ in range(n_points):
                x0 = []
                for r in ranges:
                    x0.append(np.random.uniform(r[0], r[1]))
                x0_samples.append(torch.tensor([x0], dtype=torch.float32, device=self.model.device))

        else:
            x0_samples = []
            grid_axes = [np.linspace(r[0], r[1], int(np.ceil(n_points ** (1 / len(ranges))))) for r in ranges]
            mesh = np.meshgrid(*grid_axes)
            grid_points = np.stack([m.flatten() for m in mesh], axis=-1)
            for i in range(min(n_points, grid_points.shape[0])):
                x0_samples.append(torch.tensor([grid_points[i]], dtype=torch.float32, device=self.model.device))


        for x0 in x0_samples:
            trajectory, u, converged = self.generate_trajectory(x0, t_span, time_step, min_delta, patience)
            if plot:
                self.plot_trajectory_from_data(trajectory, u, time_step)
            if converged:
                self.model.logger.info(f"Trajectory from {x0.cpu().detach().numpy()} converged.")
                success_count += 1
            else:
                self.model.logger.info(f"Trajectory from {x0.cpu().detach().numpy()} did not converge.")

        success_rate = success_count / n_points
        self.model.logger.info(f"Success rate over {n_points} trajectories: {success_rate * 100:.2f}%")
        return success_rate
    

    def plot_trajectory(self, x0: torch.Tensor, t_span: np.ndarray, time_step: float, min_delta: float = None, patience: int = None):
        trajectory, u, converged = self.generate_trajectory(x0, t_span, time_step, min_delta, patience)

    def plot_trajectory_from_data(self, trajectory: torch.Tensor, u: torch.Tensor, time_step: float):
        trajectory = trajectory.cpu().detach().numpy()
        u = u.cpu().detach().numpy()
        labels = self.model.hparams.problem_params.labels

        self.model.logger.info(f"Full trajectory shape = {trajectory.shape}")
        self.model.logger.info(f"Full u shape = {u.shape}")

        plt.figure(figsize=(10, 7))
        time = np.arange(trajectory.shape[0]) * time_step

        for i in range(trajectory.shape[1]):
            plt.plot(time, trajectory[:, i], label=labels[i])

        plt.plot(time[1:], u, label='Control Input u', color='orange', linestyle='--')

        plt.title('State Trajectories and Control Input')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_value_function(self):
        input_ranges = self.hparams.problem_params.input_ranges
        n_points = 100

        x1 = np.linspace(input_ranges[0][0], input_ranges[0][1], n_points)
        x2 = np.linspace(input_ranges[1][0], input_ranges[1][1], n_points)
        X1, X2 = np.meshgrid(x1, x2)
        inputs = np.stack([X1.ravel(), X2.ravel()], axis=1)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        
        g_x, _, _, _ = self.get_outputs(inputs_tensor)

        values = g_x.cpu().detach().numpy().reshape(X1.shape)

        if values.shape != X1.shape:
            # If output is (N, 1), squeeze to (N,)
            values = values.squeeze()
            values = values.reshape(X1.shape)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X1, X2, values, cmap='viridis')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Value')
        ax.set_title('Value Function Surface')
        plt.show()