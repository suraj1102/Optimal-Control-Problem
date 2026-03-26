import numpy as np
import torch
import pygame
from models.env import ProblemEnv
from visualizers.pendulum import PendulumVisualizer
from models.problem import problem


class PendulumEnv(ProblemEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(self, problem, time_step, max_steps, action_bounds,
                 term_radius=None, render_mode=None, scale_factor=None, **vis_kwargs):

        super().__init__(problem, time_step, max_steps, action_bounds, term_radius)

        self.render_mode = render_mode
        self._vis: PendulumVisualizer = None
        self._vis_kwargs = vis_kwargs
        self.scale_factor = scale_factor

    # ------------------------------------------------------------------
    # Lazy initialisation — Pygame window only opens on first render()
    # ------------------------------------------------------------------

    def _init_visualizer(self):
        self._vis = PendulumVisualizer(
            model=None,           # No ValueFunctionModel needed for env-only use
            problem=self.problem,
            time_step=self.dt,
            initial_state=self._state.clone(),
            **self._vis_kwargs,
        )
        # Disable the visualizer's own controller so it doesn't double-compute u
        self._vis.controller_enabled = False

    # ------------------------------------------------------------------
    # Gym API overrides
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Sync visualizer to the new initial state if it already exists
        if self._vis is not None:
            self._vis.state = self._state.clone()
            self._vis.trajectory.clear()
            self._vis.time = 0.0
            self._vis.control_input = 0.0

        return obs, info

    def step(self, action):
        if self.scale_factor is not None:
            action = action * self.scale_factor

        obs, reward, terminated, truncated, info = super().step(action)

        # Keep visualizer time in sync
        if self._vis is not None:
            self._vis.time += self.dt
            self._vis.control_input = float(action[0]) if len(action) == 1 else 0.0

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        # Open window on first call
        if self._vis is None:
            self._init_visualizer()

        # Abort if the user closed the window
        if not self._vis.running:
            return

        # --- sync env state → visualizer ---
        self._vis.state = self._state.clone()

        # Append bob position to trajectory history
        theta = self._state[0, 0].item()
        length_px = self.problem.length * self._vis.scale
        bob_x = self._vis.origin_x + length_px * np.sin(theta)
        bob_y = self._vis.origin_y - length_px * np.cos(theta)
        self._vis.trajectory.append((self._vis.time, bob_x, bob_y))
        if len(self._vis.trajectory) > self._vis.max_trajectory_length:
            self._vis.trajectory.pop(0)

        # --- one frame ---
        self._vis.handle_events(pygame.event.get())
        self._vis.draw()
        pygame.display.flip()
        self._vis.clock.tick(60)

    def close(self):
        if self._vis is not None:
            pygame.quit()
            self._vis = None