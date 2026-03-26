% Define System Dynamics
syms x1, x2, x3, x4, real
syms u1 real
syms M, m, l, g, real

% x1 - theta; x2 - theta_dot; x3 - position x, x4 - velocity_x
den = 1 / (M + m * sin(x1)^2);
l_t_dot_sq = l * x2 ^ 2;

f_x = [
    x2;
    (1 / l) * den * (-m * l_t_dot_sq * cos(x1) * sin(x1) - (M + m) * g * sin(x1));
    x4;
    den * m * sin(x1) * (l_t_dot_sq + g * sin(x1))
  ];

g_x = [
    0;  
    -(1 / l) * den * cos(x1);
    0;
    den;
  ];

% Define LQR Loss
L = x'*Q*x + u'*R*u;

% Define value function V symbolically
syms V real
assumeAlso(V, 'real')

% Define gradient of V with respect to x as a symbolic function
n = length(x);  % number of states
V_x = sym('V_x', [n, 1], 'real');  % Create symbolic gradient vector
assume(V_x, 'real');

% --- Compute Hamiltonian ---
HJB = -V_x' * (f_x + g_x * u) - L;

% --- Differentiate HJB wrt u1 and u2 ---
% Number of controls
m = length(u);

% Gradient of HJB w.r.t. u
dHJB_du = gradient(HJB, u);

% Solve first-order optimality condition
u_star = solve(dHJB_du == 0, u, 'ReturnConditions', false);

% --- Normalize output to symbolic column vector ---
if isstruct(u_star)
    % Multiple-input case returning struct
    u_star = struct2cell(u_star);
    u_star = vertcat(u_star{:});
elseif isa(u_star, 'sym')
    % Already a symbolic vector (or scalar)
    u_star = u_star(:);
else
    error("Unexpected solve() output type.");
end

u_star = simplify(u_star);

% --- Substitute u* back into HJB ---
HJB = simplify(subs(HJB, u, u_star));
HJB = simplify(expand(HJB), 'Steps', 50);
HJB = collect(HJB, V_x);

HJB = -HJB;  % match paper convention
