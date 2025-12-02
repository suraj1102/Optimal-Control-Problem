import pygame
import numpy as np
import math
import torch
from xtfc_model import ValueFunctionModel
from hparams import hparams

pygame.init()

WIDTH, HEIGHT = 800, 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (20, 20, 46)
RED = (233, 69, 96)
PINK = (255, 107, 122)
BLUE = (70, 130, 180)
GREEN = (0, 255, 136)
ORANGE = (255, 136, 0)
GRAY = (74, 74, 106)
DARK_GRAY = (42, 42, 62)

g = 9.81
l = 1.0
m = 0.2
dt = 0.01

Q = np.array([[100.0, 0.0], [0.0, 1.0]])
R = np.array([[0.1]])

class PendulumState:
    def __init__(self):
        self.theta = 1.0
        self.theta_dot = 1.0
        self.control_input = 0.0
        self.time = 0.0
        
    def reset(self):
        self.theta = 1.0
        self.theta_dot = 1.0
        self.control_input = 0.0
        self.time = 0.0

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]"""
    wrapped = ((angle + np.pi) % (2 * np.pi)) - np.pi
    return wrapped

def compute_control_input(grad_v):
    """Compute optimal control input"""
    V_x2 = grad_v[1]
    u = -V_x2 / (2 * l**2 * m * R[0, 0])
    return u

def update_simulation(state, model):
    """Update simulation state"""
    state_tensor = torch.tensor([state.theta, state.theta_dot], dtype=torch.float32)
    g_x, g_0, v, grad_v = model.get_outputs(state_tensor)
    u = compute_control_input(grad_v)
    
    f_x1 = state.theta_dot
    f_x2 = (g / l) * torch.sin(torch.tensor(state.theta)).item()
    g_x2 = 1 / (m * l * l)
    
    theta_dot_dot = f_x2 + g_x2 * u
    
    state.theta = wrap_angle(state.theta + f_x1 * dt)
    state.theta_dot = state.theta_dot + theta_dot_dot * dt
    state.control_input = u
    state.time += dt

def draw_grid(screen):
    """Draw background grid"""
    for i in range(0, WIDTH, 50):
        pygame.draw.line(screen, DARK_GRAY, (i, 0), (i, HEIGHT), 1)
    for i in range(0, HEIGHT, 50):
        pygame.draw.line(screen, DARK_GRAY, (0, i), (WIDTH, i), 1)

def draw_pendulum(screen, state, center_x, center_y, pendulum_length):
    """Draw the pendulum system"""
    # Calculate pendulum bob position
    theta_tensor = torch.tensor(state.theta)
    bob_x = center_x + pendulum_length * torch.sin(theta_tensor).item()
    bob_y = center_y - pendulum_length * torch.cos(theta_tensor).item()
    
    # Draw upright reference line (dashed)
    dash_length = 10
    for i in range(0, int(pendulum_length), dash_length * 2):
        start_y = center_y - i
        end_y = max(center_y - i - dash_length, center_y - pendulum_length)
        pygame.draw.line(screen, GRAY, (center_x, start_y), (center_x, end_y), 2)
    
    # Draw cart/base
    cart_rect = pygame.Rect(center_x - 40, center_y - 10, 80, 20)
    pygame.draw.rect(screen, GRAY, cart_rect)
    pygame.draw.rect(screen, BLUE, cart_rect, 2)
    
    # Draw control force arrow
    if abs(state.control_input) > 0.1:
        force_scale = 2
        force_x = center_x + state.control_input * force_scale
        color = GREEN if state.control_input > 0 else ORANGE
        
        # Arrow line
        # pygame.draw.line(screen, color, (center_x, center_y + 30), (force_x, center_y + 30), 3)
        
        # Arrow head
        # arrow_dir = np.sign(state.control_input)
        # arrow_points = [
        #     (force_x, center_y + 30),
        #     (force_x - arrow_dir * 10, center_y + 25),
        #     (force_x - arrow_dir * 10, center_y + 35)
        # ]
        # pygame.draw.polygon(screen, color, arrow_points)
    
    # Draw pendulum rod
    pygame.draw.line(screen, RED, (center_x, center_y), 
                    (int(bob_x), int(bob_y)), 4)
    
    # Draw pendulum bob
    pygame.draw.circle(screen, RED, (int(bob_x), int(bob_y)), 15)
    pygame.draw.circle(screen, PINK, (int(bob_x), int(bob_y)), 15, 2)
    
    # Draw pivot point
    pygame.draw.circle(screen, BLACK, (center_x, center_y), 8)
    pygame.draw.circle(screen, BLUE, (center_x, center_y), 8, 2)
    
    return int(bob_x), int(bob_y)

def draw_text_info(screen, state, font):
    """Draw simulation information"""
    texts = [
        f"θ: {state.theta:.3f} rad ({math.degrees(state.theta):.1f}°)",
        f"ω: {state.theta_dot:.3f} rad/s",
        f"u: {state.control_input:.2f} N·m",
        f"t: {state.time:.2f} s"
    ]
    
    for i, text in enumerate(texts):
        text_surface = font.render(text, True, WHITE)
        screen.blit(text_surface, (10, 10 + i * 25))

def draw_instructions(screen, font_small, paused):
    """Draw control instructions"""
    instructions = [
        "Controls:",
        "SPACE - Play/Pause",
        "R - Reset",
        "DRAG - Move pendulum",
        "",
        f"Status: {'PAUSED' if paused else 'RUNNING'}"
    ]
    
    y_offset = HEIGHT - len(instructions) * 20 - 10
    for i, text in enumerate(instructions):
        text_surface = font_small.render(text, True, WHITE)
        screen.blit(text_surface, (10, y_offset + i * 20))

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Inverted Pendulum Controller")
    clock = pygame.time.Clock()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ValueFunctionModel(in_dim=2, out_dim=1, hparams=hparams).to(device)
    model.load_state_dict(torch.load("models\inverted-pendulum_xtfc-unfreeze_[50]_SiLU.pt", map_location=device))
    model.eval()
    
    # Fonts
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 20)
    
    # Simulation state
    state = PendulumState()
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    pendulum_length = 150
    
    # Control variables
    running = True
    paused = False
    dragging = False
    
    # Simulation loop
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state.reset()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                theta_tensor = torch.tensor(state.theta)
                bob_x = center_x + pendulum_length * torch.sin(theta_tensor).item()
                bob_y = center_y - pendulum_length * torch.cos(theta_tensor).item()
                
                dist = math.sqrt((mouse_x - bob_x)**2 + (mouse_y - bob_y)**2)
                if dist < 30:
                    dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    dx = mouse_x - center_x
                    dy = mouse_y - center_y
                    
                    new_theta = math.atan2(dx, -dy)
                    state.theta = wrap_angle(new_theta)
                    state.theta_dot = 0
        
        # Update simulation
        if not paused and not dragging:
            update_simulation(state, model)
        
        # Drawing
        screen.fill(BLACK)
        draw_grid(screen)
        bob_x, bob_y = draw_pendulum(screen, state, center_x, center_y, pendulum_length)
        draw_text_info(screen, state, font)
        draw_instructions(screen, font_small, paused)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()