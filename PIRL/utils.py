from scipy.stats import qmc
import matplotlib.pyplot as plt
import torch

class manager:
    def __init__(self, env, config):
        self.env = env
        self.cfg = config
        self.state_dim = env.observation_space.shape[0]
        # scramble=True is good for removing grid artifacts in low dimensions
        self.sampler = qmc.Sobol(d=self.state_dim, scramble=True)

    def get_sobol_samples(self, n_samples):
        raw_samples = self.sampler.random(n=n_samples)
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        return low + raw_samples * (high - low)

    def collect_integral_batch(self, agent, initial_states, segment_length=10):
        # Use the dt and environment from self
        dt = self.env.unwrapped.dt
        batch_s, batch_sn, batch_a, batch_cost = [], [], [], []

        for s_start in initial_states:
            # Re-initialize to the sampled state
            self.env.reset()
            self.env.unwrapped.state = s_start
            
            s_tensor = torch.tensor(s_start, dtype=torch.float32)
            # Use the current actor to decide the action for this segment
            action = agent.actor(s_tensor).detach().numpy()
            
            integral_cost = 0
            current_s = s_start
            
            for _ in range(segment_length):
                # Calculate cost: x'Qx + u'Ru
                # Ensure agent.Q/R are tensors; converting to numpy for the math
                step_cost = (current_s @ agent.Q.numpy() @ current_s + 
                            action @ agent.R.numpy() @ action)
                
                integral_cost += step_cost * dt
                
                # Apply action to move the physics forward
                current_s, _, _, _, _ = self.env.step(action)
                
            # Store the trajectory endpoints and the integral
            batch_s.append(s_start)
            batch_sn.append(current_s)
            batch_a.append(action)
            batch_cost.append(integral_cost)

        return (torch.tensor(np.array(batch_s), dtype=torch.float32),
                torch.tensor(np.array(batch_sn), dtype=torch.float32),
                torch.tensor(np.array(batch_a), dtype=torch.float32),
                torch.tensor(np.array(batch_cost), dtype=torch.float32))