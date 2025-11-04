from imports import *

"""
TFC:
    V(x, theta) = g(x, theta) + V(0) - g(0, theta)
    g(x, theta) -> NN

Training Steps:
    Randomize and Freeze weights and biases of h1
"""

class X_TFC(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hidden_units=[128], activation=nn.Tanh):
        super().__init__()
        # Use ModuleList so submodules are registered and moved when calling .to(device)
        self.layers = nn.ModuleList()

        # Create layers
        for i, num_hidden_units in enumerate(hidden_units):
            hi = nn.Linear((in_dim if i == 0 else hidden_units[i - 1]), hidden_units[i])
            self.layers.append(hi)
        
        # Initialize layer weights and biases
        for i, layer in enumerate(self.layers):
            nn.init.uniform_(layer.weight, -1, 1)
            nn.init.uniform_(layer.bias, -1, 1)
        
        # Create output layer 
        self.y = nn.Linear(hidden_units[-1], out_dim, bias=False)
        self.activation = activation()

        # Initialize output layer | No bias is given
        nn.init.uniform_(self.y.weight, -1, 1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        
        x = self.y(x)
        return x

    def forward_layers_output(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return x

    def analytical_pretraning(self, X_init, target_func, regularization=1e-6):
        X_init = X_init.to(device)

        with torch.no_grad():
            H = self.forward_layers_output(X_init)  # [N, num_hidden_units[-1] ]
            target = target_func(X_init).to(device)  # [N, out_dim]

            print(f"Hidden layer shape: {H.shape}")
            print(f"Target shape: {target.shape}")

            # Analytical solution: W = (HᵀH + λI)⁻¹ Hᵀ T
            HTH = H.T @ H
            HTT = H.T @ target

            I = torch.eye(HTH.shape[0], device=device)
            HTH_reg = HTH + regularization * I

            try:
                beta_analytical = torch.linalg.solve(HTH_reg, HTT)  # [num_hidden_units, out_dim]

                # Copy to network weights
                self.y.weight.data.copy_(beta_analytical.T)  # PyTorch expects [out_dim, num_hidden_units]
                
                # Evaluate fit quality
                V_approx = H @ beta_analytical
                mse_error = torch.mean((V_approx - target) ** 2)

                print(f"Analytical initialization completed successfully.")
                print(f"Approximation MSE: {mse_error.item():.6e}")
                print(f"Output weights norm: {torch.norm(self.y.weight).item():.6f}")

                return True, mse_error.item()

            except Exception as e:
                print(f"Analytical solution failed: {e}")
                print("Proceeding with random initialization...")
                return False, float('inf')


def pde_residual(model, x):
    # x: (N, 2) tensor
    x = x.clone().requires_grad_(True)
    x_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=device)
    v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    
    g = model(x) # N, 1
    g_0 = model(x_bc)

    V = g + v_bc - g_0


    grads = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True)[0]
    # grads is N, 2

    V_x1 = grads[:, 0]
    V_x2 = grads[:, 1]

    x1 = x[:, 0]
    x2 = x[:, 1] 

    # PDE terms (same as before but using V from TFC)
    term1 = -0.5 * (torch.square(x1) + torch.square(x2))
    term2 = - V_x1 * x2
    term3 = 0.5 * torch.square(V_x2)
    pde_residual = term1 + term2 + term3
    boundry_residual = v_bc - g_0

    return pde_residual, boundry_residual, V, g
