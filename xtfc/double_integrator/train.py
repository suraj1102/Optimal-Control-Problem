from imports import *
from model import *

# Hyperparameters
hidden_units = [128]
activation = nn.Tanh
n_colloc = 10_000
lr = 1e-3
n_epochs = 100_000 + 1

def main():
    model = X_TFC(in_dim=2, out_dim=1, hidden_units=hidden_units, activation=activation).to(device)

    # Freeze layer weights and biases
    for layer in model.layers:
        for p in layer.parameters():
            p.requires_grad = False

    # Ensure output layer is trainable
    for p in model.y.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable}/{total}") 

    x_init = sample_inputs(n_sample=2000).to(device)

    target_func = lambda x: 0.5 * torch.square(x[:, 0:1] + x[:, 1:2])
    model.analytical_pretraning(x_init, target_func)

    # Since weights and biases of all layers are frozen, no need to use model.parameters()
    optimizer = optim.Adam(model.y.parameters(), lr=lr)
    
    # tranining loop
    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        x_colloc = sample_inputs(n_sample=n_colloc)
        x_colloc.requires_grad_(True)

        residual, V_out = pde_residual(model, x_colloc)
        pde_loss = torch.mean(residual**2)
        
        loss = pde_loss
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} | Total Loss: {loss.item():.4e}"
            )

    # Save model
    torch.save(model.state_dict(), "x_tfc_di.pth")

if __name__ == '__main__':
    main()