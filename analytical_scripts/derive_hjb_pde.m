clear; clc; close;

% Define symbolic variables
syms x1 x2 u1 real
x = [x1; x2];
u =  u1;

% ------ PARAMETERS -------
problem = "inverted-pendulum";

if problem == "nonlinear-dynamics"
    Q = [1 0; 0 1];
    R = 1;

    f_x = [
        x2 - x1;
        -x1/2 - x2/2 + x1^2 * x2 / 2
    ];

    g_x = [
        0;
        x1
    ];

elseif problem == "nonlinear-dynamics-2"
    Q = [1 0; 0 1];
    R = 1;

    f_x = [
        -x1 + x2;
        -x1/2 - x2/2 * (1 - (cos(2*x1) + 2)^2)
    ];

    g_x = [
        0;
        cos(2*x1) + 2
    ];

elseif problem == "double-integrator"
    Q = 1/2 * [1 0; 0 1];
    R = 1/2 * 1;

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
    Q = [1 0; 0 0.1];
    R = 1;

    % g = 9.81;
    % l = 1;
    % m = 1;

    syms g l m real

    f_x = [
        x2;
        g/l * sin(x1)
    ];

    g_x = [
        0;
        1 / (m*l*l)
    ];

else
    error("Unknown problem identifier: " + problem);
end


% ---------- SCRIPT ------------

% Define LQR Loss
L = x'*Q*x + u*R*u;

% Define value function V symbolically
syms V real
assumeAlso(V, 'real')

% Define gradient of V with respect to x as a symbolic function
n = length(x);  % number of states
V_x = sym('V_x', [n, 1], 'real');  % Create symbolic gradient vector
assume(V_x, 'real');

% Define Hamilton-Jacobi-Bellman (HJB) equation
HJB = -V_x' * (f_x + g_x * u) - L;


% Differentiate HJB with respect to control input u
dHJB_du = real(diff(HJB, u));


% Solve for optimal control input u*
u_star = solve(dHJB_du == 0, u, 'ReturnConditions', false);

% Substitute optimal control input into HJB equation
HJB = simplify(subs(HJB, u, u_star));
HJB = simplify(expand(HJB), 'Steps', 50); % Computes the products
HJB = collect(HJB, V_x); % Groups by the terms 

% ------------ End of script --------------
% The rest is just to clean the output

% Remove real() wrappers from x1, x2, and V_x entries
for i = 1:length(x)
    HJB = subs(HJB, real(x(i)), x(i));
    eval(['syms V_x' num2str(i) ' real']);
    HJB = subs(HJB, real(diff(V, x(i))), eval(['V_x' num2str(i)]));
    u_star = subs(u_star, real(x(i)), x(i));
    u_star = subs(u_star, real(diff(V, x(i))), eval(['V_x' num2str(i)]));
end

HJB = -HJB; % Stored in this format in paper
HJB_latex = latex(HJB);
disp(u_star);
disp(latex(u_star));
disp(HJB);
disp(HJB_latex);