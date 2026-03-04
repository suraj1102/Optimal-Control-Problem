clc; clear; close all;

syms x1 x2 x3 x4 u real
syms m mc l g real

M = mc + m;
J = m*l^2;

theta = x1;
omega = x2;

den = M*(m*l^2 + J) - (m*l*cos(theta))^2;

alpha = M*m*g*l*sin(theta) - (m*l*cos(theta))*(u + m*l*omega^2*sin(theta));
alpha = alpha / den;

f_x = [
    x2;
    (M*m*g*l*sin(theta) + (m*l*cos(theta))*(m*l*omega^2*sin(theta))) / den;
    x4;
    (-alpha*cos(theta) + omega*sin(theta))*m*l/M
];

g_x = [
    0;
    (-m*l*cos(theta))/den;
    0;
    1/M
];

xdot = f_x + g_x*u;


x = [x1 x2 x3 x4]';

A = jacobian(xdot, x);
B = jacobian(xdot, u);

eq = [x1 x2 x3 x4 u];
eq_val = [0 0 0 0 0];

A = subs(A, eq, eq_val);
B = subs(B, eq, eq_val);

% Example parameters
m_val  = 1;
mc_val = 4;
l_val  = 1;
g_val  = 10;

A = subs(A, ...
    [x1 x2 x3 x4 u m mc l g], ...
    [0  0  0  0  0 m_val mc_val l_val g_val]);

B = subs(B, ...
    [x1 x2 x3 x4 u m mc l g], ...
    [0  0  0  0  0 m_val mc_val l_val g_val]);

A = double(A);
B = double(B);

A = double(A);
B = double(B);

Q = [1 0 0 0;
     0 0.1 0 0;
     0 0 1 0;
     0 0 0 0.1];

R = 1;


[K,S,e] = lqr(A,B,Q,R);

disp(S)