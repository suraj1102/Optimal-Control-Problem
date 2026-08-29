import torch


class PIRL:
    def __init__(self, env, config):

        # Make Q, R tensors
        Q = config["cost_matrices"]["Q"]
        R = config["cost_matrices"]["R"]
        Q = torch.diag(torch.tensor(Q))
        R = torch.diag(torch.tensor(R))

        self.Q = Q
        self.R = R
        self.env = env
        self.config = config

        constraint = config['system_params']['constraint']
        self.instantialize_neural_nets(constraint)
        self.instantiate_optimizers()

    def instantialize_neural_nets(self, constraint):
        from NeuralNets import Actor, Critic, AdmissibleNet, ActorConstraint

        if constraint:
            self.actor = ActorConstraint(self.env, self.config)
        else:
            self.actor = Actor(self.env, self.config)

        self.critic = Critic(self.env, self.config)
        # self.admissible_net = AdmissibleNet(self.env, self.config)

    def instantiate_optimizers(self):
        lr_a = self.config["hyperparameters"]["lr_actor"]
        lr_c = self.config["hyperparameters"]["lr_critic"]
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_c)


class Algo1(PIRL):
    def __init__(self, env, config, system_dynamics_model):
        super().__init__(env, config)
        self.F = system_dynamics_model


class Algo2(PIRL):
    def __init__(self, env, config):
        super().__init__(env, config)
        # TODO: Algo 2
        raise NotImplementedError()
