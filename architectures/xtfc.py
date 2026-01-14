from utils import *
from models.hparams import hparams

"""
TFC:
    V(x, theta) = g(x, theta) + V(0) - g(0, theta)
    g(x, theta) -> NN
"""

class ValueFunctionModel(nn.Module):
    def __init__(self, in_dim=2, out_dim=1, hparams=hparams):
        super().__init__()
        self.hparams = hparams
        hidden_units = hparams['hidden_units']
        self.x_bc = torch.zeros((1, in_dim), dtype=torch.float32, device=device)
        self.v_bc = torch.tensor([[0.0]], dtype=torch.float32, device=device)

        # Use ModuleList so submodules are registered and moved when calling .to(device)
        self.layers: nn.ModuleList = nn.ModuleList()

        # Create layers
        for i, num_hidden_units in enumerate(hidden_units):
            hi = nn.Linear((in_dim if i == 0 else hidden_units[i - 1]), hidden_units[i])
            self.layers.append(hi)
        
        # Initialize layer weights and biases
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, nn.Linear) # To get rid of LSP error 
            nn.init.uniform_(layer.weight, -1, 1)
            nn.init.uniform_(layer.bias, -1, 1)
        
        # Create output layer 
        self.y = nn.Linear(hidden_units[-1], out_dim, bias=True if 'bias' in hparams['architecture'] else False)
        self.activation = hparams['activation']()

        # Initialize output layer | No bias is given
        nn.init.uniform_(self.y.weight, -1, 1)

    def forward(self, x):
        x = self.forward_layers_output(x)
        x = self.y(x)
        return x

    def forward_layers_output(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
        return x

    def freeze_parameters(self, which_params: str):
        """
        Freezes the parameters of the model based on the specified parameter group.
        
        Args:
            which_params (str): Specifies which parameters to freeze. Options are:
                - 'layers': Freeze all layers except the output layer.
                - 'all': Freeze all parameters including the output layer.
                - 'output': Freeze only output parameters.
        """
        if which_params in ['layers', 'all']:
            # Freeze layer weights and biases
            for layer in self.layers:
                for p in layer.parameters():
                    p.requires_grad = False

        if which_params in ['output', 'all']:
            # Freeze output layer
            for p in self.y.parameters():
                p.requires_grad = False

        if which_params == 'layers':
            # Ensure output layer is trainable
            for p in self.y.parameters():
                p.requires_grad = True

    def unfreeze_parameters(self, which_params: str):
        """
        Unfreezes the parameters of the model based on the specified parameter group.
        
        Args:
            which_params (str): Specifies which parameters to unfreeze. Options are:
                - 'layers': Unfreeze all layers except the output layer.
                - 'all': Unfreeze all parameters including the output layer.
                - 'output': Unfreeze only output parameters.
        """
        if which_params in ['layers', 'all']:
            # Unfreeze layer weights and biases
            for layer in self.layers:
                for p in layer.parameters():
                    p.requires_grad = True

        if which_params in ['output', 'all']:
            # Unfreeze output layer
            for p in self.y.parameters():
                p.requires_grad = True

        if which_params == 'layers':
            # Ensure output layer remains frozen
            for p in self.y.parameters():
                p.requires_grad = False

    def get_outputs(self, x: torch.Tensor):
        x.requires_grad_(True)
        

        g_x = self(x)  # N, 1
        g_0 = self(self.x_bc)

        v = g_x + self.v_bc - g_0

        grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        # grad_v is N, 2 | 2 is in_dim

        return g_x, g_0, v, grad_v

    def xTQx_analytical_pretraning(self, target_func, n_sample=2000, regularization=1e-6):
        print(target_func)
        x_init = sample_inputs(n_sample=2000, dim=hparams['in_dim'], edge_weight=hparams['edge_sampling_weight'], input_range=hparams['input_range']).to(device)

        with torch.no_grad():
            H = self.forward_layers_output(x_init)  # [N, num_hidden_units[-1] ]
            target = target_func(x_init).to(device)  # [N, out_dim]

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

    def LQR_analytical_pretraining(self, system_dynamics):
        pass # TODO