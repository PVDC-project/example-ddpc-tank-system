%% Four tank system DDMPC
% Berberich et al.: Data-Driven Model Predictive Control with Stability and Robustness Guarantees
% https://arxiv.org/pdf/1906.04679.pdf
% Berberich et al.: Data-Driven Tracking MPC for Changing Setpoints
% https://arxiv.org/pdf/1910.09443.pdf
% Raff et al: Nonlinear MPC of a Four Tank System
% https://ieeexplore.ieee.org/abstract/document/4776652
% Coulson et al.: Data-Enabled Predictive Control: In the Shallows of the DeePC
% https://arxiv.org/pdf/1811.05890.pdf
clear;clc;close all;

%% Define the model
Ts = 1;  % [s] sampling time
A = [0.921,0,0.041,0;...
     0,0.918,0,0.033;...
     0,0,0.924,0;...
     0,0,0,0.937];
B = [0.017,0.001;...
     0.001,0.023;...
     0,0.061;...
     0.072,0];
C = [1,0,0,0;
     0,1,0,0];
nx = size(A,1); nu = size(B,2); ny = size(C,1);  % system dimensions
sys = ss(A,B,C,0,Ts);  % object for simulation

%% Create the PE trajectory and Hankel matrices
T_test = 100;  % [s] duration of the test trajectory
N = T_test/Ts;  % number of samples in the test trajectory
L = 25;  % length of the prediction horizon

% create the test input
rng(10)  % fix random number generator seed
t_test = 0:Ts:(N-1)*Ts;  % time vector
u_min = -1; u_max = 1;  % input limits
ud = u_min + (u_max-u_min)*rand(nu,N);  % uniform random input, columns are samples

xd = zeros(nx,1);  % initial condition
yd = nan(ny,N);  % columns are samples
for i=1:N
    yd(:,i) = C*xd;  % test output, from the same time step as inputs
    xd = A*xd + B*ud(:,i);  % state evolution, x+=Ax+Bu
end

% plot the trajectory
test_trajectory = figure; movegui(test_trajectory,[1200 60]);
subplot(3,1,1); plot(t_test, yd(1,:)); ylabel('y1 [m]'); title('PE trajectory')
subplot(3,1,2); plot(t_test, yd(2,:)); ylabel('y2 [m]')
subplot(3,1,3); plot(t_test, ud(1,:)); hold on; plot(t_test, ud(2,:));
ylabel('control input'); xlabel('Time [s]')

% create the Hankel matrices of order L+nx, check for persistency of excitation
Hu = HankelMatrix(ud,L+nx);
if (rank(Hu) ~= nu*(L+nx)); warning('Input is not PE'); end
Hy = HankelMatrix(yd,L+nx);
H = [Hu;Hy];

%% Create the controller
yalmip('clear')
u = sdpvar(nu,L+nx,'full');  % total input sequence
y = sdpvar(ny,L+nx,'full');  % total output sequence
u0 = sdpvar(nu,nx,'full');  % past n inputs (measurements)
y0 = sdpvar(ny,nx,'full');  % past n outputs (measurements)
ys = sdpvar(ny,1);  % artificial equilibrium output
us = sdpvar(nu,1);  % artificial equilibrium input
yT = sdpvar(ny,1);  % output target 
uT = sdpvar(nu,1);  % input target 
alpha = sdpvar(size(H,2),1);  % (N-L+1)x1 vector, H*alpha=[u;y]
% slack = sdpvar(ny*nx,1);  % slack variable for constraint relaxation

% controller parameters
Q = 5*eye(ny);  % output weights
R = 1*eye(nu);  % input weights
S = 0*eye(nu);  % terminal input weight
T = 200*eye(ny);  % terminal output weight
lambda_alpha = 1e-4;  % regularization term on alpha

% objective function
objective = 0;
for k = nx+1:nx+L  % running cost, y'Qy + u'Ru, first nx elements are for the initial condition
    objective = objective + (y(:,k)-ys)'*Q*(y(:,k)-ys) + (u(:,k)-us)'*R*(u(:,k)-us);
end
objective = objective + (us-uT)'*S*(us-uT);  % target input offset cost
objective = objective + (ys-yT)'*T*(ys-yT);  % target output offset cost
objective = objective + lambda_alpha * norm(alpha)^2;  % regularization cost

% constraints
u_max = [1.2; 2];  u_min = -u_max;  % input limits
y_max = 1.2; y_min = 0;  % output limits
constraints = H*alpha == [u(:);y(:)];  % the model of the data-driven MPC
constraints = [constraints, u(:,1:nx)==u0(:,1:nx), y(:,1:nx)==y0(:,1:nx)];  % initial state constraint (through n last measurements)
constraints = [constraints, u_min(1) <= u(1,nx+1:L+nx) <= u_max(1)];  % input box constraints
constraints = [constraints, u_min(2) <= u(2,nx+1:L+nx) <= u_max(2)];  % input box constraints
constraints = [constraints, y_min <= y(:,nx+1:L) <= y_max];  % output box constraints
constraints = [constraints, u(:,L+1:L+nx)==repmat(us,1,nx), y(:,L+1:L+nx)==repmat(ys,1,nx)];  % terminal constraint
constraints = [constraints, 0.99*u_min <= us <= 0.99*u_max];  % equilibrium input limits
constraints = [constraints, 0.99*y_min <= ys <= 0.99*y_max];  % equilibrium output limits

% define the controller
parameters_in = {yT,u0,y0};  % reference and the current "state"
solutions_out = {u,y,alpha};  % optimal input, output and "model"
ops = sdpsettings('verbose',0);  % print output? (0-2)
controller = optimizer(constraints,objective,ops,parameters_in,solutions_out);

%% Closed-loop simulation
% Initial conditions
x_sim = zeros(nx,1);  % initial state
u0_sim = zeros(nu,nx);  % initial past input measurements
y0_sim = C*repmat(x_sim,1,nx);  % initial past output measurements

% Reference signal
yT_sim = [0.65; 0.77];  % system equilibrium

% Simulation
simplot = figure; movegui(simplot,[1200 570]);  % for live plotting
Tsim = 400;  % [s], simulation time
Nsim = Tsim/Ts;  % number of simulation steps
uhist = nan(nu,Nsim); yhist = nan(ny,Nsim); yThist = nan(ny,Nsim);  % for logging
for i = 1:Nsim
    
    if i > 150  % step time
        yT_sim = 2*[0.5;0.5];
    end

    inputs = {yT_sim,u0_sim,y0_sim};
    [solutions, errorcode, errortext, ~, ~, diagnostics] = controller{inputs};  
    
    if errorcode ~= 0
        error(errortext{1});
    else
        U = solutions{1};  % optimal inputs
        Y = solutions{2};  % optimal outputs
        Alpha = solutions{3};  % optimal alpha
    end

    % shift initial condition, new input and output
    u0_sim(:,1:end-1) = u0_sim(:,2:end);
    u0_sim(:,end) = U(:,nx+1);
    y0_sim(:,1:end-1) = y0_sim(:,2:end);
    y0_sim(:,end) = C*x_sim;
    
    % perform one simulation step    
    x_sim = A*x_sim + B*U(:,nx+1);    

    % signal history
    uhist(:,i) = U(:,nx+1);
    yhist(:,i) = y0_sim(:,end);
    yThist(:,i) = yT_sim;

    % live plots
    figure(simplot)

    subplot(2,2,1);cla;hold on;
    stairs((i:i+size(U,2)-nx-1)*Ts,U(1,nx+1:end)','b');  % predicted control input
    stairs((1:i)*Ts,uhist(1,1:i)','g');  % applied control input
    yline([u_min(1) u_max(1)],'r--')
    title('Control input 1')

    subplot(2,2,3);cla;hold on;
    stairs((i:i+size(U,2)-nx-1)*Ts,U(2,nx+1:end)','b');  % predicted control input
    stairs((1:i)*Ts,uhist(2,1:i)','g');  % applied control input
    yline([u_min(2) u_max(2)],'r--')
    xlabel('Time [s]');
    title('Control input 2')

    subplot(2,2,2);cla;hold on;
    stairs((i-nx:i-nx+size(Y,2)-1)*Ts,Y(1,1:end),'b');  % predicted system output
    stairs((1:i)*Ts,yhist(1,1:i),'g')  % system output
    stairs((1:i)*Ts,yThist(1,1:i),'r--')  % reference
    title('System output 1')

    subplot(2,2,4);cla;hold on;
    stairs((i-nx:i-nx+size(Y,2)-1)*Ts,Y(2,1:end),'b');  % predicted system output
    stairs((1:i)*Ts,yhist(2,1:i),'g')  % system output
    stairs((1:i)*Ts,yThist(2,1:i),'r--')  % reference
    xlabel('Time [s]');
    title('System output 2')

    pause(0.01) 
end

%% Auxiliary functions
function H = HankelMatrix(x,L)
% A function for constructing the Hankel matrix of size(nx*L,N-L+1) of a data
% sequence of size nx x N (columns are individual samples).
nx = size(x,1);
N = size(x,2);
H = nan(nx*L,N-L+1);

X = x(:);  % concatenate data to a single vector
for i = 1:N-L+1  % iterate over columns
    H(:,i) = X((i-1)*nx+1:(i-1)*nx+L*nx);
end
end