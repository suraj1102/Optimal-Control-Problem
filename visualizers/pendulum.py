import pygame
import numpy as np
import torch
from models.pygame_visualizer import PygameVisualizer


class PendulumVisualizer(PygameVisualizer):
    def __init__(self, model, problem, width=800, height=600, time_step=0.01, initial_state=None):
        super().__init__(model, width, height, "Pendulum Visualizer")
        self.problem = problem
        
        self.dt = time_step
        self.time = 0.0
        self.controller_enabled = True
        
        self.state = torch.tensor([[np.pi + 0.1, 0.0]], dtype=torch.float32, device=model.device) if initial_state is None else initial_state.to(model.device)
        
        # Trajectory history for plotting
        self.trajectory = []
        self.max_trajectory_length = 50
        
        # Visualization parameters
        self.scale = 150  # Pixels per meter
        self.origin_x = width // 2
        self.origin_y = height // 2
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.rod_color = (100, 150, 255)
        self.bob_color = (255, 100, 100)
        self.trajectory_color = (100, 255, 100)
        self.pivot_color = (200, 200, 200)
        self.text_color = (255, 255, 255)
        
        # Control parameters
        self.control_input = 0.0
        self.paused = False
        self.show_trajectory = False
        
        # Font for displaying information
        self.font = pygame.font.Font(None, 24)
        
    def reset(self, theta=np.pi, theta_dot=0.0):
        """Reset the pendulum to a new initial state."""
        self.state = torch.tensor([[theta, theta_dot]], dtype=torch.float32, device=self.model.device)
        self.trajectory = []
        self.time = 0.0
        self.control_input = 0.0
        
    def compute_control(self):
        if not self.controller_enabled:
            return 0.0

        state_for_grad = self.state.clone().detach().requires_grad_(True)
        _, _, _, grad_v = self.model.get_outputs(state_for_grad)
        
        u = self.problem.control_input(self.state, grad_v)
        
        return u.item()
    
    def update_dynamics(self, u):
        f_x = self.problem.f_x(self.state)
        g_x = self.problem.g_x(self.state)
        
        state_dot = f_x + g_x * u
        
        self.state = self.state + state_dot * self.dt
        
        # Normalize theta to [-pi, pi]
        self.state[0, 0] = torch.atan2(torch.sin(self.state[0, 0]), torch.cos(self.state[0, 0]))
        
        self.time += self.dt
        
    def draw(self):
        self.screen.fill(self.bg_color)
        
        # Get current state
        theta = self.state[0, 0].item()
        theta_dot = self.state[0, 1].item()
        
        # Calculate pendulum positions
        length_pixels = self.problem.length * self.scale
        bob_x = self.origin_x + length_pixels * np.sin(theta)
        bob_y = self.origin_y - length_pixels * np.cos(theta)
        
        # Draw trajectory
        if self.show_trajectory and len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                _, x1, y1 = self.trajectory[i]
                _, x2, y2 = self.trajectory[i + 1]
                alpha = int(255 * (i / len(self.trajectory)))
                color = (*self.trajectory_color, alpha)
                pygame.draw.line(self.screen, self.trajectory_color, (x1, y1), (x2, y2), 2)
        
        # Draw pivot point
        pygame.draw.circle(self.screen, self.pivot_color, (self.origin_x, self.origin_y), 8)
        
        # Draw rod
        pygame.draw.line(self.screen, self.rod_color, 
                        (self.origin_x, self.origin_y), 
                        (int(bob_x), int(bob_y)), 4)
        
        # Draw bob
        bob_radius = int(self.problem.mass * 20)
        pygame.draw.circle(self.screen, self.bob_color, (int(bob_x), int(bob_y)), bob_radius)
        
        # Draw reference line (upright position)
        pygame.draw.line(self.screen, (50, 50, 50), 
                        (self.origin_x, self.origin_y), 
                        (self.origin_x, self.origin_y - length_pixels), 1)
        
        # Display information
        info_texts = [
            f"Time: {self.time:.2f} s",
            f"Theta: {theta:.3f} rad ({np.degrees(theta):.1f}°)",
            f"Theta_dot: {theta_dot:.3f} rad/s",
            f"Control u: {self.control_input:.3f}",
            f"",
            f"Controls:",
            f"SPACE: Pause/Resume",
            f"R: Reset to upright",
            f"D: Reset to downward",
            f"T: Toggle trajectory",
            f"Arrow Keys: Perturb state",
            f"Z: Toggle controller (currently {'ON' if self.controller_enabled else 'OFF'})"
        ]
        
        y_offset = 10
        for text in info_texts:
            surface = self.font.render(text, True, self.text_color)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
        
        # Show paused status
        if self.paused:
            pause_surface = self.font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(pause_surface, (self.width - 100, 10))
    
    def handle_events(self, events):
        """Handle keyboard and window events."""
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    
                elif event.key == pygame.K_r:
                    # Reset to upright position with small perturbation
                    self.reset(theta=0.1, theta_dot=0.0)
                    
                elif event.key == pygame.K_d:
                    # Reset to downward position
                    self.reset(theta=np.pi + 0.1, theta_dot=0.0)
                    
                elif event.key == pygame.K_t:
                    self.show_trajectory = not self.show_trajectory
                    
                elif event.key == pygame.K_UP:
                    # Increase angular velocity
                    self.state[0, 1] += 0.5
                    
                elif event.key == pygame.K_DOWN:
                    # Decrease angular velocity
                    self.state[0, 1] -= 0.5
                    
                elif event.key == pygame.K_LEFT:
                    # Rotate counterclockwise
                    self.state[0, 0] -= 0.1
                    
                elif event.key == pygame.K_RIGHT:
                    # Rotate clockwise
                    self.state[0, 0] += 0.1

                elif event.key == pygame.K_z:
                    # Toggle controller
                    self.controller_enabled = not self.controller_enabled
    
    def update(self):
        """Update the simulation state."""
        if not self.paused:
            # Compute control input from the model
            self.control_input = self.compute_control()
            
            # Update dynamics
            self.update_dynamics(self.control_input)
            
            # Store trajectory
            theta = self.state[0, 0].item()
            length_pixels = self.problem.length * self.scale
            bob_x = self.origin_x + length_pixels * np.sin(theta)
            bob_y = self.origin_y - length_pixels * np.cos(theta)
            
            self.trajectory.append((self.time, bob_x, bob_y))
            
            # Limit trajectory length
            if len(self.trajectory) > self.max_trajectory_length:
                self.trajectory.pop(0)