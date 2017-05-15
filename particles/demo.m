%% Particle Filter
%  for Switching Linear Dynamic Systems (SLDS):
%  z_t ~ P(z_t|z_{t-1})
%  x_t = A(z_t)x_{t-1} + B(z_t)w_t + F(z_t)u_t
%  y_t = C(z_t)x_t + D(z_t)v_t + G(z_t)u_t
%  PF and RBPF are based on software by Nando de Freitas

clear all; close all;

%% model parameters

N = 128;                    % Number of particles.
T = 64;                     % Number of time steps.

n_x = 1;                    % Dimension of hidden state.
n_z = 4;                    % Number of discrete states.
n_y = 2;                    % Dimension of observations.

resampling = 2;             %1: residual, 2: deterministic, 3: multinomial

%% generate data

[par, x, z, y, u] = slds(N,T,n_x,n_z,n_y);

%% particle filter

[z_pf, x_pf, w_pf, time_pf] = pf(N,T,n_x,n_z,n_y,par,y,u,resampling);

[z_rbpf, w_rbpf, time_rbpf] = rbpf(N,T,n_x,n_z,n_y,par,y,u,resampling);

%% generate plots

%SLDS
subplot(311)
plot(1:T,z,'r','linewidth',2);
ylabel('z_t','fontsize',15);
axis([0 T+1 0 n_z+1])
grid on;
subplot(312)
plot(1:T,x,'r','linewidth',2);
ylabel('x_t','fontsize',15);
grid on;
subplot(313)
plot(1:T,y,'r','linewidth',2);
ylabel('y_t','fontsize',15);
xlabel('t','fontsize',15);
grid on;

%compute most likely state
z_plot_pf = zeros(T,N);
z_plot_rbpf = zeros(T,N);
for t=1:T,
  z_plot_pf(t,:) = z_pf(1,t,:);
  z_plot_rbpf(t,:) = z_rbpf(1,t,:);
end;

z_num_pf = zeros(T,n_z);
z_num_rbpf = zeros(T,n_z);
z_max_pf = zeros(T,1);
z_max_rbpf = zeros(T,1);
for t=1:T,
  for i=1:n_z,
    z_num_pf(t,i)= length(find(z_plot_pf(t,:)==i));
    z_num_rbpf(t,i)= length(find(z_plot_rbpf(t,:)==i));
  end;
  [arb,z_max_pf(t)] = max(z_num_pf(t,:));  
  [arb,z_max_rbpf(t)] = max(z_num_rbpf(t,:));  
end;

%MAP estimate
figure;
plot(1:T,z,'k',1:T,z,'ko',1:T,z_max_rbpf,'r+',1:T,z_max_pf,'bv','linewidth',1);
legend('True state','RBPF MAP estimate','PF MAP estimate');
axis([0 T+1 0.5 n_z+0.5])

%display detection errors and time
detect_error_pf   = sum(z~=z_max_pf');
detect_error_rbpf = sum(z~=z_max_rbpf');

fprintf('PF detection errors: %d\n', detect_error_pf);
fprintf('RBPF detection errors: %d\n', detect_error_rbpf);

fprintf('PF time: %s sec\n', num2str(time_pf));
fprintf('RBPF time: %s sec\n', num2str(time_rbpf));

