%% Simulated Cart Pole Dynamics

clc; clear; close all;

% Initial condition
% x1 - theta; x2 - theta_dot; x3 - position x, x4 - velocity_x
x0 = [0; 0; 0; 0];
tspan = [0 10];

p.mc = 4;
p.m = 5;
p.l = 1;
p.g = 10;

% Control input
u = 1;

% ODE Solver
RHS = @(t,x) cartpole_dynamics2(t, x, u, p);
sol = ode45(RHS, tspan, x0);


%% Animation
l = p.l;
speedupFactor = 3;   % 1 = real time, 2 = 2x faster, 0.5 = slow motion
figure;
grid on
axis equal

xmin = min(sol.y(3, :)) - 0.1 * max(min(sol.y(3, :)), 5);
xmax = max(sol.y(3, :)) + 0.1 * max(max(sol.y(3, :)), 5);
axis([xmin xmax -0.2 1])
hold on

cart = rectangle('Position',[0 0 0.6 0.2], 'FaceColor', 'k');
pole = plot([0 0],[0 0],'c','LineWidth',2);
bob = plot(0,0,'bo','MarkerSize',10,'MarkerFaceColor','b');
drawnow

tic
t = 0;
while t <= tspan(2)
    % Use deval to interpolate ode45 solution at time t
    states_now = deval(sol, t);

    theta  = states_now(1);
    cart_x = states_now(3);

    % Pole tip
    pole_x = cart_x + l*sin(theta);
    pole_y = l*cos(theta);

    % Cart Position 
    
    set(cart,'Position',[cart_x-0.3 -0.1 0.6 0.2]);
    set(pole,'XData',[cart_x pole_x], 'YData',[0 pole_y]);
    set(bob,'XData',pole_x, 'YData',pole_y);
    
    drawnow limitrate
    t = toc;
end

%% Function Definitions
function xdot = cartpole_dynamics(t, x, u, p)
    % Equations from https://underactuated.mit.edu/acrobot.html#cart_pole
    mc = p.mc;
    m = p.m;
    l = p.l;
    g = p.g;
    
    % x1 - theta; x2 - theta_dot; x3 - position x, x4 - velocity_x
    theta = x(1);
    theta_dot = x(2);
    pos = x(3);
    vel = x(4);
    
    den = 1 / (mc + m * sin(theta)^2);
    l_t_dot_sq = l * theta_dot^2;
    
    A = [
        theta_dot;
        (1 / l) * den * (-m * l_t_dot_sq * cos(theta) * sin(theta) - (mc + m) * g * sin(theta));
        vel;
        den * m * sin(theta) * (l_t_dot_sq + g * sin(theta))
    ];
    
    B = [
        0;  
        -(1 / l) * den * cos(theta);
        0;
        den;
    ];
    
    xdot = A + B * u;
end

function zdot = cartpole_dynamics2(t, z, u, p)
    % From https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
    mc = p.mc;
    m = p.m;
    l = p.l;
    g = p.g;
    M = mc + m;

    J = m*l^2;

    % x1 - theta; x2 - theta_dot; x3 - position x, x4 - velocity_x
    theta = z(1);
    omega = z(2);
    x = z(3);
    vx = z(4);


    alpha = M*m*g*l*sin(theta) - (m*l*cos(theta)) * (u + m*l*omega^2*sin(theta));
    alpha = alpha / ( M*(m*l^2+J) - (m*l*cos(theta))^2  );

    ax = u - m*l*alpha*cos(theta) + m*l*omega^2*sin(theta);
    ax = ax / M;
    

    zdot = [omega; alpha; vx; ax];
end
