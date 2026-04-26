import torch
import torch.nn as nn
from utils import sample_states
from PIRL import Algo1, Algo2


def init_weights(m):
    """
    Initialize Weights for NN
    Usage: nnObject.apply(init_weights)
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class Algo1Trainer:
    def __init__(self, agent: Algo1):
        self.F = agent.F
        self.env = agent.env
        self.config = agent.config
        self.actorNN: nn.Module = agent.actor
        self.criticNN: nn.Module = agent.critic
        self.Q = agent.Q
        self.R = agent.R
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.kmax = int(self.config.get("kmax", 10))
        self.epsilon_v = float(self.config.get("epsilon_v", 1e-2))
        self.epsilon_u = float(self.config.get("epsilon_u", 1e-2))
        self.Nepochs = int(self.config.get("Nepochs", 5_000))
        self.batch_size = int(self.config.get("batch_size", 2**12))
        self.print_every = int(self.config.get("print_every", 200))
        self.opt_actor = agent.opt_actor
        self.opt_critic = agent.opt_critic
        self.k = 0
        self.actor_losses, self.critic_losses = [], []
        self.L_v = torch.tensor(float("inf"), device=self.device)
        self.L_u_prev = torch.tensor(float("inf"), device=self.device)

        # Send tensors to device
        self.actorNN.to(self.device)
        self.criticNN.to(self.device)
        self.Q = agent.Q.to(self.device)
        self.R = agent.R.to(self.device)

    def initialize_critic(self):
        with torch.no_grad():
            n_init = 2048
            x_init = sample_states(self.env, n_init).to(self.device)
            y_init = torch.einsum("bi,ij,bj->b", x_init, self.Q, x_init).unsqueeze(1)
        self.criticNN.train()
        init_optimizer = torch.optim.Adam(self.criticNN.parameters(), lr=1e-2)
        for init_iter in range(self.Nepochs):
            idx = torch.randint(0, n_init, (256,))
            x_batch = x_init[idx]
            y_batch = y_init[idx]
            pred = self.criticNN(x_batch)
            loss = torch.nn.functional.mse_loss(pred, y_batch)
            init_optimizer.zero_grad()
            loss.backward()
            init_optimizer.step()
            if init_iter % self.print_every == 0:
                print(f"[INIT CRITIC]: iter={init_iter}, fit xTQx loss={loss:.6f}")

    def initialize_actor(self):
        self.criticNN.eval() 
        
        for epoch_iter in range(self.Nepochs):
            states = sample_states(self.env, self.batch_size).to(self.device)
            states.requires_grad_(True)
            
            V = self.criticNN(states)
            grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
            
            actions = self.actorNN(states)
            Fxu = self.F(states, actions) 
            
            xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
            uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
            
            hamiltonian = xQx + uRu + (grad_V * Fxu).sum(dim=1)
            loss_actor = hamiltonian.mean()
            
            # backprop
            self.opt_actor.zero_grad()
            loss_actor.backward()
            self.opt_actor.step()

            if epoch_iter % self.print_every == 0:
                print(f"[INIT ACTOR] iter={epoch_iter}, Hamiltonian={loss_actor.item():.6f}")

        self.L_u_prev = loss_actor

    def policy_evaluation(self):
        self.criticNN.train()
        self.actorNN.eval()

        self.L_v = torch.tensor(float("inf"), device=self.device)
        while self.L_v.item() > self.epsilon_v:
            self.criticNN.apply(init_weights)
            for epoch_iter in range(self.Nepochs):
                states = sample_states(self.env, self.batch_size).to(self.device)
                states.requires_grad_(True)
                with torch.no_grad():
                    actions = self.actorNN(states)
                Fxu = self.F(states, actions)

                # print(f"{states.shape=}\t{actions.shape=}\t{Fxu.shape=}")
                # print(f"{states=}\n{actions=}\n{Fxu=}")

                V = self.criticNN(states).squeeze()
                grad_V = torch.autograd.grad(
                    V.sum(), states, create_graph=True, retain_graph=True
                )[0]

                # print(f"{V.shape=}\t{grad_V.shape=}")
                # print(f"{V=}\n{grad_V=}")

                xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
                uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
                lv_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
                self.L_v = (lv_terms**2).mean()
                self.opt_critic.zero_grad()
                self.L_v.backward()
                self.opt_critic.step()
                self.critic_losses.append(float(self.L_v.item()))
                if epoch_iter % self.print_every == 0:
                    print(
                        f"[Phase 1] k={self.k} iter={epoch_iter} | L_v={self.L_v.item():.6f}"
                    )

    def policy_improvement(self):
        self.actorNN.train()
        self.criticNN.eval()
        for p in self.criticNN.parameters():
            p.requires_grad = False

        L_u = torch.tensor(float("inf"), device=self.device)
        self.actorNN.apply(init_weights)

        for epoch_iter in range(self.Nepochs):
            states = sample_states(self.env, self.batch_size).to(self.device)
            states.requires_grad_(True)
            
            actions = self.actorNN(states)
            Fxu = self.F(states, actions)

            V = self.criticNN(states).squeeze()
            grad_V = torch.autograd.grad(V.sum(), states, create_graph=True)[0]
            
            xQx = torch.einsum("bi,ij,bj->b", states, self.Q, states)
            uRu = torch.einsum("bi,ij,bj->b", actions, self.R, actions)
            
            lu_terms = xQx + uRu + (grad_V * Fxu).sum(dim=1)
            L_u = lu_terms.mean()
            
            # backprop
            self.opt_actor.zero_grad()
            L_u.backward()
            self.opt_actor.step()
            
            self.actor_losses.append(float(L_u.item()))
            if epoch_iter % self.print_every == 0:
                print(
                    f"[Phase 2] k={self.k} iter={epoch_iter} | L_u={L_u.item():.6f} | L_u_prev={self.L_u_prev.item():.6f}"
                )

            if self.L_u_prev.item() < float("inf") and torch.abs(L_u - self.L_u_prev).item() <= self.epsilon_u:
                break

        self.L_u_prev = L_u

        for p in self.criticNN.parameters():
            p.requires_grad = True
        self.criticNN.train()

    def run(self):
        self.initialize_critic()
        self.initialize_actor()
        while self.k < self.kmax:
            self.policy_evaluation()
            self.policy_improvement()
            self.k += 1
            print(
                f"End of iteration k={self.k}, L_v={self.L_v.item():.6f}, L_u={self.L_u_prev.item():.6f}"
            )
        return self.actor_losses, self.critic_losses


def trainAlgo2(agent: Algo2):
    raise NotImplementedError()
