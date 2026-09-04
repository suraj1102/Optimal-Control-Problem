# Optimal Control of Continuous-Time Systems: Direct HJB Solving versus Reinforcement Learning

Source code for the course projects of **AI3001: Deep Learning** (Monsoon 2025) and
**RO3004: Reinforcement Learning** (Spring 2026), Plaksha University.
Team: Arnav Kapoor, Meher Sidhu, Suraj Dayma.

---

## Abstract

We study the infinite-horizon regulator problem for the torque-driven inverted
pendulum and ask whether embedding the Hamilton–Jacobi–Bellman (HJB) physics into a
learning objective is competitive with solving the HJB partial differential equation
(PDE) directly. Four solver families are implemented on a common environment: a
physics-informed neural network using the Extreme Theory of Functional Connections
(X-TFC); the two physics-informed reinforcement learning (PIRL) algorithms of Wang and
Wu; five model-free deep RL algorithms (PPO, A2C, DDPG, TD3, SAC); and an open-loop
trajectory-optimization benchmark. The direct X-TFC solve is the cheapest to set up,
the fastest to train, and lower in steady-state error than PIRL and most model-free
baselines. Two modifications to the X-TFC solver are reported here that were not in the
earlier version of this work: initialization of the value network from the Riccati
(LQR) solution rather than from $\mathbf{x}^\top Q \mathbf{x}$, and unfreezing the
hidden weights after the least-squares stage so the network trains end-to-end. 

---

## 1. Problem formulation

### 1.1 System

The state is $\mathbf{x} = [\theta, \dot\theta]^\top$, with $\theta$ measured from the
upright ($\theta = 0$ is inverted) and wrapped to $[-\pi, \pi]$. The scalar control is
the base torque $u = \tau$. With length $l$, mass $m$, damping coefficient $b$, and
gravity $g$, the dynamics are control-affine:

$$
\dot{\mathbf{x}} \;=\; f(\mathbf{x}) + g(\mathbf{x})\,u,
\qquad
f(\mathbf{x}) = \begin{bmatrix} \dot\theta \\[2pt] \dfrac{g}{l}\sin\theta - \dfrac{b}{m}\dot\theta \end{bmatrix},
\qquad
g(\mathbf{x}) = \begin{bmatrix} 0 \\[2pt] \dfrac{1}{ml^2} \end{bmatrix}.
$$

The undamped variant ($b = 0$) is in `problems/inverted_pendulum.py`; the damped
variant used for the reported X-TFC runs is in `problems/damped_inverted_pendulum.py`.

### 1.2 Cost and the HJB equation

The running cost is quadratic,

$$
\mathcal{L}(\mathbf{x}, u) = \mathbf{x}^\top Q\,\mathbf{x} + u^\top R_u\, u,
\qquad Q \succeq 0,\; R_u \succ 0,
$$

and the value function is the optimal cost-to-go
$V^*(\mathbf{x}(t)) = \min_u \int_0^\infty \mathcal{L}\,\mathrm{d}t$. Splitting the
integral over a short interval $\Delta t$ and letting $\Delta t \to 0$ gives the
infinite-horizon HJB PDE

$$
0 = \min_{u}\Big\{ \mathcal{L}(\mathbf{x}, u) + \nabla_{\mathbf{x}} V^*(\mathbf{x})^\top \dot{\mathbf{x}} \Big\}.
$$

For control-affine dynamics and quadratic cost the inner minimization is closed-form.
Setting $\partial/\partial u\left[u^\top R_u u + \nabla_{\mathbf{x}}V^\top (f + gu)\right] = 0$
yields the optimal feedback law

$$
u^* = -\tfrac{1}{2} R_u^{-1} g(\mathbf{x})^\top \nabla_{\mathbf{x}} V(\mathbf{x}),
$$

implemented in `problem.control_input`. Substituting $u^*$ back eliminates $u$ and
leaves a nonlinear PDE in $V$ alone,

$$
\mathbf{x}^\top Q \mathbf{x}
\;+\; \nabla_{\mathbf{x}} V^\top f(\mathbf{x})
\;-\; \tfrac{1}{4}\, \nabla_{\mathbf{x}} V^\top g(\mathbf{x}) R_u^{-1} g(\mathbf{x})^\top \nabla_{\mathbf{x}} V
\;=\; 0,
\qquad V(\mathbf{0}) = 0 .
$$

The left-hand side is the residual that the physics-informed methods drive to zero. For
the damped pendulum it expands (see `damped_inverted_pendulum.pde_residual`) to

$$
r(\mathbf{x}) =
q_{11}\theta^2 + q_{22}\dot\theta^2
+ \dot\theta\, V_\theta
- \frac{V_{\dot\theta}^2}{4 m^2 l^4 r_{11}}
+ \left(\frac{g}{l}\sin\theta - \frac{b}{m}\dot\theta\right) V_{\dot\theta}.
$$

Retaining a finite horizon $T$ instead of taking $\Delta t \to 0$ gives the integral
(Bellman) form, which is what the model-free PIRL variant solves.

---

## 2. Methods

### 2.1 Direct HJB solving with X-TFC

`architectures/xtfc.py` trains a network $V_\theta$ to make $r(\mathbf{x})$ vanish at
collocation points sampled uniformly from the state box. Two structural choices follow
the X-TFC recipe.

**Exact boundary embedding.** Rather than penalizing $V(\mathbf{0}) = 0$, the
constraint is satisfied by construction. Writing the raw network output as
$g_\text{net}$, the value is formed as

$$
V(\mathbf{x}) \;=\; g_\text{net}(\mathbf{x}) \;+\; V_{\mathrm{bc}} \;-\; g_\text{net}(\mathbf{x}_{\mathrm{bc}}),
$$

with $\mathbf{x}_{\mathrm{bc}} = \mathbf{0}$ and $V_{\mathrm{bc}} = 0$
(`XTFC.get_outputs`). This holds identically for *any* weights, which is the property
that makes Section 2.3 safe.

**Extreme Learning Machine stage.** Hidden weights and biases are drawn once from
$\mathcal{U}(-1, 1)$ and frozen; only the linear output layer $\beta$ is fitted. With
hidden features $H = \sigma(\cdot)$ and a target $T$, $\beta$ is the ridge solution

$$
\beta = \left(H^\top H + \lambda I\right)^{-1} H^\top T ,
$$

solved once in closed form (`set_weights_from_target`, $\lambda = 10^{-6}$). This is a
*pretraining* target, not the PDE: it places the network in a sensible basin before
gradient descent on the PDE residual begins.

The boundary term $V_{\mathrm{bc}} - g_\text{net}(\mathbf{0})$ is still computed and
logged as `boundary_loss`, but it is not added to the optimized loss — the embedding
already makes it zero. Only `pde_loss` is backpropagated.

### 2.2 New: Riccati (LQR) initialization of the value network

The earlier version initialized the ELM output layer to reproduce the running-cost
quadratic, $T = \mathbf{x}^\top Q \mathbf{x}$ (`perform_xTQX`). That target has the
right shape at the origin but the wrong curvature: it is the *cost rate*, not the
*cost-to-go*, and it carries no information about the dynamics or the control
authority.

The replacement (`perform_LQR_pretraining`) initializes from the value function of the
linearized problem. `_linearize_and_solve_care` linearizes the closed-loop dynamics at
the equilibrium using `torch.autograd.functional.jacobian`,

$$
A = \left.\frac{\partial \dot{\mathbf{x}}}{\partial \mathbf{x}}\right|_{\mathbf{x}_{eq}, u_{eq}},
\qquad
B = \left.\frac{\partial \dot{\mathbf{x}}}{\partial u}\right|_{\mathbf{x}_{eq}, u_{eq}},
$$

solves the continuous-time algebraic Riccati equation (CARE) with
`scipy.linalg.solve_continuous_are`,

$$
A^\top S + S A - S B R_u^{-1} B^\top S + Q = 0,
$$

and uses the quadratic form of the Riccati matrix as the least-squares target:

$$
T(\mathbf{x}) = \mathbf{x}^\top S\, \mathbf{x} \;\approx\; V^*(\mathbf{x}) \quad \text{near } \mathbf{x} = \mathbf{0}.
$$

The LQR gain $K = R_u^{-1} B^\top S$ is returned alongside and is what seeds the
admissible policy in the PIRL policy-iteration loop (Section 2.4).

Because $\mathbf{x}^\top S \mathbf{x}$ is the *exact* value function of the linearized
problem, the initialization is correct to second order at the origin rather than merely
correctly signed. The initialization target is selected by the
`hyper_params.analytical_pretraining` key (`"lqr"` or `"xTQx"`).

The routine is wrapped in a retry loop: up to `init_limit` least-squares fits are
attempted, stopping early when the fit MSE falls below `initialization_cutoff`,
otherwise keeping the best. **This loop is weaker than it looks — see Section 8.1.**

### 2.3 New: unfreezing the X-TFC weights

`architectures/xtfc_unfreeze.py` subclasses `XTFC` and overrides the single hook that
freezes the basis:

```python
class XTFC_Unfreeze(XTFC):
    def pre_train_step(self):
        pass  # do not freeze the hidden layers
```

In the base `XTFC`, `pre_train_step` calls `freeze_hidden()`, which sets
`requires_grad = False` on every hidden layer, leaving only the output layer `y`
trainable — the standard ELM condition. Overriding it to a no-op means that after the
closed-form ridge initialization of $\beta$, Adam updates *all* parameters against the
PDE residual for the full run.

The sequence is therefore:

1. Draw random hidden weights $W, b \sim \mathcal{U}(-1,1)$.
2. Solve the ridge system once for $\beta$ against the Riccati target (Section 2.2).
3. Train all of $\{W, b, \beta\}$ for `n_epochs` steps of Adam on `pde_loss`.

Two consequences follow directly from the code:

- **The boundary condition survives.** The embedding in Section 2.1 subtracts
  $g_\text{net}(\mathbf{0})$ at every forward pass, so $V(\mathbf{0}) = 0$ holds no
  matter how the hidden weights move. Unfreezing cannot violate it.
- **The X-TFC guarantee does not survive.** Once $H$ is no longer fixed, $\beta$ is no
  longer the exact least-squares optimum of anything, and the fast one-shot solve is
  reduced to an initialization. The method becomes a conventional PINN with a
  physically motivated starting point. The claimed benefit is generalization —
  a fixed random basis of 50 SiLU features is a hard capacity ceiling — and the cost is
  the loss of the convexity and speed argument that motivates X-TFC in the first place.

Note that `set_optimizer_scheduler()` constructs Adam over `self.parameters()` *before*
`pre_train_step()` runs. In the frozen case this is harmless (frozen parameters receive
no gradient and Adam skips them); in the unfrozen case every parameter is in the
optimizer as intended. The `l1_lambda` / `l2_lambda` penalties in `train_model` sum over
*all* parameters, so with unfreezing they now regularize the feature basis as well as
the output layer. Both are set to `0` in the shipped configuration.

### 2.4 Physics-informed RL (PIRL)

Two algorithms from Wang and Wu, in `RLEnv/PIRL/`.

**PIRL-1 (model-based policy iteration)** alternates two supervised fits. Policy
evaluation fits the critic to the HJB residual under the current policy,

$$
\mathcal{L}_v = \frac{1}{N_v}\sum_i \left( \mathbf{x}_i^\top Q \mathbf{x}_i + u_i^\top R_u u_i + \frac{\partial V_\theta}{\partial \mathbf{x}} F(\mathbf{x}_i, u_i) \right)^2 ,
$$

and policy improvement minimizes the Hamiltonian over the actor weights,

$$
\mathcal{L}_u = \frac{1}{N_u}\sum_i \left( \mathbf{x}_i^\top Q \mathbf{x}_i + \pi_i^\top R_u \pi_i + \frac{\partial V_\theta}{\partial \mathbf{x}} F(\mathbf{x}_i, \pi_i) \right).
$$

The loop must be seeded with an admissible (stabilizing, finite-cost) policy, obtained
from the same LQR solve as Section 2.2: $u = -K\mathbf{x}$. A constrained variant
applies $u = u_{\max}\tanh(\text{actor}(\mathbf{x}))$.

**PIRL-2 (model-free integral policy iteration)** solves the integral HJB form in an
actor–critic framework with losses designed so that the learned value function acts as
a control Lyapunov function (CLF). Over a segment $[t_0, t_0 + t_s]$ it forms integral
Hamiltonians $H_u$ and $H_v$ and penalizes $\overline{\mathrm{ReLU}}(\cdot)$ terms that
fire whenever the candidate fails to decrease. Its initialization phase learns an
admissible policy by penalizing a *quadratic* Lyapunov candidate
$W = \mathbf{x}^\top P \mathbf{x}$, $P \succ 0$.

### 2.5 Baselines

PPO, A2C, DDPG, TD3 and SAC via Stable-Baselines3 on a custom Gymnasium environment
(`RLEnv/invertedpendulum.py`); these ignore the HJB physics entirely. The benchmark is
an open-loop optimal trajectory computed by direct nonlinear programming
(`RLEnv/comparison/`).

---

## 3. Experimental setup

### 3.1 X-TFC solver (`yamls/unfreeze_ip.yaml`)

| Setting | Value |
|---|---|
| Problem | damped inverted pendulum |
| $g,\ l,\ m,\ b$ | $10,\ 1.0,\ 0.1,\ 0.1$ |
| $Q,\ R_u$ | $\mathrm{diag}(2.0,\ 0.1)$, $[0.5]$ |
| State box | $\theta \in [-0.5, 0.5]$, $\dot\theta \in [-2.0, 2.0]$ |
| Architecture | 1 hidden layer, 50 units, SiLU, no output bias |
| Collocation points / step | 5000, resampled every epoch |
| Epochs | 10 000, Adam, lr $10^{-3}$ |
| Pretraining | LQR target, 1000 points, $\lambda = 10^{-6}$, cutoff $10^{-6}$, ≤100 attempts |
| $\ell_1 / \ell_2$ | 0 / 0 |

Collocation points are drawn i.i.d. uniform over the box on every training step
(`sample_inputs`), so the residual is minimized in expectation over the box rather than
on a fixed grid.

### 3.2 RL experiments

$\mathrm{d}t = 0.02$ s, 1000-step episodes, torque bounded to $[-1, 1]$, $b = 0.1$,
$g = 10$, $l = 0.8$, $m = 0.1$, initial states uniform on $[-0.5, 0.5]^2$, termination
when $|\theta|$ or $|\dot\theta|$ leaves $(\pi, 10)$; 42 parallel environments and
$10^6$ steps per run. Three reward definitions are compared — quadratic (the negative
of $\mathcal{L}$), a survival reward ($+1$ inside a threshold angle, $-0.1$ outside),
and a cosine reward $R = q_1\cos\theta - q_2\dot\theta^2 - r_1 u^2$ — under both a
no-disturbance baseline and a combined Gaussian-plus-impulse schedule. The PIRL runs
use $g = 10$, $l = 1$, $m = 0.1$, $b = 0.2$, integration step 0.05 s, segment length
10, Sobol sampling of 5000 collocation states, and $u_{\max} = 0.7$ for the constrained
variant.

---

## 4. Results

### 4.1 Cross-method comparison

Steady-state error, mean $|\theta|$ over the last 100 steps of a 1000-step rollout from
$\mathbf{x}_0 = [0.1, 1.0]$. Smaller is better; trajectory optimization is the
benchmark.

| Method | Type | $\|\theta\|$ (rad) |
|---|---|---|
| Trajectory optimization | benchmark (open-loop) | 0.0000 |
| SAC | model-free RL | 0.0066 |
| X-TFC (direct HJB) | physics-informed PINN | 0.0111 |
| DDPG | model-free RL | 0.0169 |
| PIRL-1 | physics-informed RL | 0.0532 |
| TD3 | model-free RL | 0.0974 |

TD3's error of $\approx 0.097$ rad ($5.6^\circ$) is an order of magnitude worse than the
best methods. X-TFC is the closest of the physics-informed methods and tracks the
benchmark's running cost most closely. PIRL-1 lands between the best and worst,
consistent with it being close to a copy of its LQR initialization.

### 4.2 Reward shape

Under the quadratic cost, learning was by far the most reliable. The survival reward
makes every state inside the threshold band equal, so agents converge but settle with a
steady-state error inside the band; at a wide $12^\circ$ threshold they oscillate rather
than settle. The cosine reward is gameable: some agents zero the velocity in a region
where $\cos\theta > 0$ without ever reaching upright. The quadratic cost penalizes the
true distance to the origin and admits neither degenerate optimum.

### 4.3 PIRL-1: the policy iteration buys little

In the unconstrained case PIRL-1 almost reproduces the LQR-initialized policy, changing
it only slightly, and its value function coincides with the convex value function from
the direct X-TFC solve — the outer policy-iteration loop adds little where the value
function is convex and the initialization is already close. Adding the $\tanh$
constraint produces a choppy, discontinuous action function and a value function that is
no longer convex. These outcomes are stable in the number of iterations, persisting from
5 up to 25 (about two hours of training).

### 4.4 PIRL-2: negative result

PIRL-2 assumes the following.

> **Assumption (quadratic admissibility certificate).** There exist
> $W(\mathbf{x}) = \mathbf{x}^\top P \mathbf{x}$ with $P \succ 0$ and a policy
> $\pi^{(0)}$ such that $\dot W < 0$ along the closed-loop dynamics over a meaningful
> region of state space.

For the inverted pendulum, whose upright equilibrium is a saddle of the uncontrolled
dynamics, no quadratic $W$ satisfies this over a region large enough to certify
PIRL-2's initial policy. A quadratic Lyapunov function is a single bowl centred at the
origin; away from the small linearized neighbourhood the gravitational term
$(g/l)\sin\theta$ pushes the state away from upright, so no quadratic bowl decreases
along the flow over a meaningful region. Because PIRL-2's initialization is trained
precisely by penalizing violations of this certificate, the initialization phase cannot
succeed and the algorithm does not converge.

To confirm that the cause is the unstable equilibrium and not an implementation defect,
the same code was run on an inherently stable nonlinear system,

$$
\dot x_1 = -x_1 + x_2, \qquad
\dot x_2 = -\tfrac{1}{2}x_1 - \tfrac{1}{2}x_2 + \tfrac{1}{2}x_1^2 x_2 + x_1 u,
$$

for which a quadratic Lyapunov function exists over a meaningful region. There PIRL-2
converges and produces a stabilizing rollout. This is a limitation of the quadratic
certificate, not of the control problem: the other three families all stabilize the
pendulum.

---

## 5. Repository layout

Two halves sharing one repository.

**Repository root — direct HJB solving with PINNs**

- `main.py` — entry point; pretrains, trains, rolls out the controller, writes
  `trajectory_output.csv`
- `architectures/` — `pinn.py` (soft boundary penalty), `xtfc.py` (embedded boundary +
  frozen ELM basis), `xtfc_unfreeze.py` (Section 2.3)
- `models/` — `valuefunctionmodel.py` (base module, training loop),
  `hparams.py` (dataclass config, YAML loader), `problem.py` (abstract problem,
  optimal control law), `simulator.py`, `pygame_visualizer.py`
- `problems/` — pendulum, damped pendulum, cart-pole, double integrator, nonlinear system
- `yamls/` — run configurations (`pinn.yaml`, `unfreeze.yaml`, `unfreeze_ip.yaml`)
- `analytical_scripts/` — MATLAB derivations of the HJB residual and the dynamics
- `visualizers/` — pygame pendulum viewer

**`RLEnv/` — reinforcement learning** (see `RLEnv/README.md`)

- `baseenv.py`, `invertedpendulum.py` — shared Gymnasium environment
- `experiment.py` — Stable-Baselines3 runs (PPO, A2C, DDPG, TD3, SAC)
- `training/` — callbacks, reward and disturbance definitions, rollout evaluation
- `PIRL/` — the two physics-informed RL algorithms; `main.py` is the entry point
- `comparison/` — trajectory-optimization benchmark and cross-method plots
- `policy_table/`, `EmbeddedPendulum/` — hardware deployment

---

## 6. Setup and running

The two halves have separate environments.

```bash
# PINN / HJB side (repository root)
conda env create -f conda_env.yml      # or: pip install -r requirements.txt

# RL side
conda env create -f RLEnv/environment.yml -n rlenv
```

```bash
python main.py                         # solve the HJB PDE with a PINN
python RLEnv/experiment.py             # train the SB3 baselines
python RLEnv/PIRL/main.py              # train PIRL-1 / PIRL-2
```

`RLEnv/comparison/compare.ipynb` reproduces the comparison figures and Table 4.1.

### 6.1 Configuring an X-TFC run

The YAML controls the numerics; the architecture and problem classes are selected in
`main.py`. The keys that matter for the two contributions above:

```yaml
hyper_params:
  analytical_pretraining: "lqr"    # "lqr" (Section 2.2) | "xTQx" (previous) | "none"

pretraining_params:
  n_pretraining_colloc: 1000       # points used for the ridge fit
  lambda_reg: 0.000001             # ridge regularization
  initialization_cutoff: 0.000001  # accept a fit below this MSE; -1 disables the loop
  init_limit: 100                  # maximum fit attempts
```

To switch between the frozen and unfrozen solver, change the class instantiated in
`main.py`:

```python
model = XTFC(problem)            # frozen basis: standard X-TFC
model = XTFC_Unfreeze(problem)   # all weights trainable after the ridge init
```

---

## 7. Hardware deployment

The learned policy runs on a physical pendulum: a generic DC motor with an optical
encoder driving a 105 g arm of effective length 13.5 cm. Commanding torque directly is
impractical without the motor constants, so the policy is converted offline to position
control through a lookup table. The state space is discretized
($\theta \in [-\pi, \pi]$, $\dot\theta \in [-2, 2]$, both at resolution 0.015) and for
each grid point the next angle $\theta_{t+\mathrm{d}t}$ produced by the policy after
$\mathrm{d}t = 0.01$ s of simulation is recorded, giving triples
$[\theta_t, \dot\theta_t, \theta_{t+\mathrm{d}t}]$. On the microcontroller, state
estimation runs on each encoder interrupt, the position target is refreshed at 100 Hz,
and an inner PID position loop runs ten times faster at 1 kHz. Storing the table as
integer offsets in flash (PROGMEM) rather than floating point cuts policy-storage memory
by roughly 75 %, saves on the order of 10 KB of RAM, and reduces inference to a direct
lookup.

The main practical limitation is that the high-level policy is discretized and
interpolated on the grid, with closed-loop feedback supplied by the inner PID loop
rather than by re-evaluating the policy continuously.

---

## References

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks,"
   *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.
2. R. Furfaro, A. D'Ambrosio, E. Schiassi, and A. Scorsoglio, "Physics-informed neural
   networks for closed-loop guidance and control in aerospace systems," *AIAA SCITECH
   2022 Forum*.
3. T. R. Srinivasa and S. Kumar, "Solving infinite-horizon optimal control problems using
   the extreme theory of functional connections," *11th Indian Control Conference (ICC)*,
   2025. arXiv:2510.27187. [IEEE](https://ieeexplore.ieee.org/document/11372426)
4. Y. Wang and Z. Wu, "Physics-informed reinforcement learning for optimal control of
   nonlinear systems," *AIChE Journal*, vol. 70, no. 10, e18542, 2024.
   [Link](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.18542)
5. D. Mortari, "The theory of connections: Connecting points," *Mathematics*, vol. 5,
   no. 4, p. 57, 2017.
6. G.-B. Huang, Q.-Y. Zhu, and C.-K. Siew, "Extreme learning machine: Theory and
   applications," *Neurocomputing*, vol. 70, no. 1–3, pp. 489–501, 2006.
7. E. Schiassi *et al.*, "Extreme theory of functional connections: A fast
   physics-informed neural network method for solving ordinary and partial differential
   equations," *Neurocomputing*, vol. 457, pp. 334–356, 2021.
8. M. Lutter, B. Belousov, K. Listmann, D. Clever, and J. Peters, "HJB optimal feedback
   control with deep differential value functions and action constraints," *CoRL*, 2020.
9. A. Raffin *et al.*, "Stable-Baselines3," *JMLR*, vol. 22, no. 268, pp. 1–8, 2021.
10. M. Kelly, "An introduction to trajectory optimization," *SIAM Review*, vol. 59,
    no. 4, pp. 849–904, 2017.
