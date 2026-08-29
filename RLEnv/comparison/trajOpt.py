import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

OUTPUT_DIR = "outputs"
FILENAME = "trajopt_rollout"
SAVEPATH = os.path.join(OUTPUT_DIR, FILENAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# parameters
dt = 0.01
tf = 10
N = int(tf / dt)

length = 1
m = 0.1
g = 10
b = 0.1

# initial state
theta0 = 0.1
thetadot0 = 1

# cost
Q = np.diag([2, 0.1])
R = np.diag([0.5])

# casadi variables
theta = ca.MX.sym('theta')
thetadot = ca.MX.sym('thetadot')
x = ca.vertcat(theta, thetadot)
u = ca.MX.sym('u')

# dynamics
thetaddot = g/length * ca.sin(theta) - (b/m)*thetadot + u/(m*length**2)
xdot = ca.vertcat(thetadot, thetaddot)

# euler discretisation
x_next = x + dt * xdot
f = ca.Function('f', [x, u], [x_next])

# list of variables to give NLP
X = ca.MX.sym('X', 2, N+1)
U = ca.MX.sym('U', 1, N)

# objective
cost = 0
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    cost += ca.mtimes([xk.T, Q, xk]) + ca.mtimes([uk.T, R, uk])

# constraints
g = []
g.append(X[:,0] - ca.vertcat(theta0, thetadot0))

for k in range(N):
    x_next_pred = f(X[:,k], U[:,k])
    g.append(X[:,k+1] - x_next_pred)

g = ca.vertcat(*g)

# NLP
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
nlp = {'x': opt_vars, 'f': cost, 'g': g}

solver = ca.nlpsol('solver', 'ipopt', nlp)

# solve
sol = solver(lbg=0, ubg=0)

opt = sol['x'].full().flatten()

# extract
X_opt = opt[:2*(N+1)].reshape((N+1, 2)).T
U_opt = opt[2*(N+1):].reshape((1, N))

thetas = X_opt[0, :]
thetadots = X_opt[1, :]
controls = np.concatenate([U_opt[0, :], [0]])  # pad for same length

# reward (negative cost per step)
rewards = -(thetas**2 * Q[0,0] + thetadots**2 * Q[1,1] + controls**2 * R[0,0])

# save csv
df = pd.DataFrame({
    'theta': thetas,
    'thetadot': thetadots,
    'control': controls,
    'reward': rewards
})
df.to_csv(f'{OUTPUT_DIR}/{FILENAME}.csv', index=False)

# plot
fig, axs = plt.subplots(4, 1, figsize=(6, 8), sharex=True)

axs[0].plot(thetas)
axs[0].set_ylabel('theta')
axs[0].grid(True)

axs[1].plot(thetadots)
axs[1].set_ylabel('thetadot')
axs[1].grid(True)

axs[2].plot(controls)
axs[2].set_ylabel('control')
axs[2].grid(True)

axs[3].plot(rewards)
axs[3].set_ylabel('reward')
axs[3].set_xlabel('Timestep')
axs[3].grid(True)

fig.suptitle(f'{FILENAME} Results')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{FILENAME}.png")
plt.show()