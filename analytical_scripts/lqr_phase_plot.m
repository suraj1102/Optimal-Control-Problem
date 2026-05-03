% inverted pendulum (fixed pivot) LQR on nonlinear plant

clear; clc;

% parameters
m = 0.1;     % mass
l = 1;     % length
g = 10;

% state: x = [theta; theta_dot]

% linearized system around theta=0
A = [0 1;
     g/l 0];

B = [0;
     1/(m*l^2)];

% LQR weights
Q = diag([1 0.1]);
R = 0.01;

% LQR gain
K = lqr(A,B,Q,R);

% nonlinear dynamics
f = @(t,x) nonlinear_dynamics(x, K, m, l, g);

tspan = [0 10];

% sweep initial angles
theta0_list = linspace(-pi, pi, 10);

figure
hold on;

for i = 1:length(theta0_list)
    x0 = [theta0_list(i); 0];
    
    [t,x] = ode45(f, tspan, x0);
    
    plot(x(:,1), x(:,2));
end

xlabel('\theta (rad)');
ylabel('\theta_dot (rad/s)');
title('Phase plot (nonlinear pendulum with LQR)');
grid on;


% --- nonlinear dynamics ---
function dx = nonlinear_dynamics(x, K, m, l, g)

    th = wrapToPi(x(1));
    thd = x(2);
    
    umax = 1;
    u = -K * [th; thd];
    u = max(min(u, umax), -umax);
    
    % nonlinear dynamics
    thdd = (g/l)*sin(th) + u/(m*l^2);
    
    dx = [thd;
          thdd];
end

% --- value function surface V = x' S x ---

S = care(A,B,Q,R);  % solution of Riccati equation

theta = linspace(-pi, pi, 100);
thetad = linspace(-10, 10, 100);

[TH, THD] = meshgrid(theta, thetad);

V = zeros(size(TH));

for i = 1:size(TH,1)
    for j = 1:size(TH,2)
        x = [TH(i,j); THD(i,j)];
        V(i,j) = x' * S * x;
    end
end

figure
surf(TH, THD, V)
xlabel('\theta (rad)')
ylabel('\theta_dot (rad/s)')
zlabel('V = x^T S x')
title('LQR Value Function Surface')
grid on