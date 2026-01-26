clear; clc; close;

% Define symbolic variables
syms x1 x2 x3 x4 u1 u2 real
x = [x1; x2];
u =  u1;

% ------ PARAMETERS -------
problem = "inverted-pendulum";

if problem == "nonlinear-dynamics"
    syms q11 q22 real
    syms r11 real

    Q = [q11 0;
        0 q22
        ];

    R = [r11];

    f_x = [
        x2 - x1;
        -x1/2 - x2/2 + x1^2 * x2 / 2
    ];

    g_x = [
        0;
        x1
    ];

elseif problem == "nonlinear-dynamics-2"
    syms q11 q22 real
    syms r11 real

    Q = [q11 0;
        0 q22
        ];

    R = [r11];

    f_x = [
        -x1 + x2;
        -x1/2 - x2/2 * (1 - (cos(2*x1) + 2)^2)
    ];

    g_x = [
        0;
        cos(2*x1) + 2
    ];

elseif problem == "double-integrator"
    syms q11 q22 real
    syms r11 real

    Q = [q11 0;
        0 q22
        ];

    R = [r11];

    f_x = [
        x2;
        0
    ];

    g_x = [
        0;
        1
    ];

elseif problem == "inverted-pendulum"
    % x1 -> theta | x2 -> dot(theta)
    syms q11 q22 real
    syms r11 real

    Q = [q11 0;
        0 q22
        ];

    R = [r11];

    syms g l m real

    f_x = [
        x2;
        g/l * sin(x1)
    ];

    g_x = [
        0;
        1 / (m*l*l)
    ];

elseif problem == "damped-inverted-pendulum"
    % x1 -> theta | x2 -> dot(theta)
    syms q11 q22 real
    syms r11 real
    syms gamma real

    Q = [q11 0;
        0 q22
        ];

    R = [r11];

    syms g l m real

    f_x = [
        x2;
        g/l * sin(x1) - (gamma / m) * x2
    ];

    g_x = [
        0;
        1 / (m*l*l)
    ];


elseif problem == "double-input-cart-pole"
    syms q11 q22 q33 q44 real
    syms r11 r22 real

    Q = [q11 0 0 0;
        0 q22 0 0;
        0 0 q33 0;
        0 0 0 q44
        ];

    R = [r11 0;
        0 r22];

    syms g l m_p m_c y real

    f_x = [
        x2;
        0;
        x4;
        -(g/l) * sin(x3);
    ];

    g_x = [
        0 0;
        1/m_c 0;
        0 0;
        -y/(m_p * l*l) -1/(m_p * l*l)
    ];

    x = [x1 x2 x3 x4]';
    u = [u1 u2]';
    

else
    error("Unknown problem identifier: " + problem);
end


% ---------- SCRIPT ------------

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
HJB = collect(HJB, V_x)

HJB = -HJB;  % match paper convention



% Generate LaTeX
HJB_latex = latex(HJB);
 
% Display results
disp('Optimal control u*:');
disp(u_star);
 
disp('u* (LaTeX):');
disp(latex(u_star));
 
disp('HJB PDE:');
disp(HJB);
 
disp('HJB (LaTeX):');
disp(HJB_latex);
