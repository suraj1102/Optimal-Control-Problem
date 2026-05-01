clc; close all; clear;

% System Parameters
g = 10;
l = 0.8;
m = 0.1;

A = [0, 1; 
     g/l, 0];
B = [0; 
     1/(m*l^2)];

A, B

% Cost Matrices
Q = diag([1.0, 0.1]);
R = 0.01;

[K,S,P] = lqr(A, B, Q, R);

% Display Result
disp('LQR Gain Matrix K:');
disp(K);


% Grid
theta_vec = linspace(-pi, pi, 50);
thetadot_vec = linspace(-4, 4, 50);
[TH, THDOT] = meshgrid(theta_vec, thetadot_vec);

% Compute Control Input u = -K * x
U = -(K(1)*TH + K(2)*THDOT);

% Plotting
figure;
surf(TH, THDOT, U);
xlabel('\theta (rad)');
ylabel('\theta_{dot} (rad/s)');
zlabel('Control Input u');
title('LQR Control Law u = -Kx');
colorbar;
shading interp;
grid on;


% Compute Value Function x^TSx
V = S(1,1).*TH.^2 + (S(1,2) + S(2,1)).*TH.*THDOT + S(2,2).*THDOT.^2;

% Plotting
figure;
surf(TH, THDOT, V);
xlabel('\theta (rad)');
ylabel('\theta_{dot} (rad/s)');
zlabel('V(x)');
title('Value Function');
colorbar;
shading interp;
grid on;