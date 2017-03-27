%% Kalman Filter and Smoother
%  applied to Linear Dynamic System (LDS):
%  z_n = Az_{n-1} + w_n, where w_n ~ N(0,G)
%  x_n = Cz_n + v_n,     where v_n ~ N(0,S)
%  z_1 = mu_0 + u,       where u ~ N(0, V0)

clear; close all;
rng('default');

%% generate data
d = 2;  % data dimension
k = 4;  % latent variable dimension
n = 1e1; % number of data points
[X,Z,model] = ldsRnd(d,k,n);

%% kalman filter
[mu, V, llh] = kalmanFilter(X, model);

%% kalman smoother
[nu, U, Ezz, Ezy, llh] = kalmanSmoother(X, model);

%% tracking error
df = X(1:2,:) - mu(1:2,:); err_filt = norm(df,2)
ds = X(1:2,:) - nu(1:2,:); err_smooth = norm(ds,2)

%% generate plots
figure;
hold on; grid on;
plot(X(1,:), X(2,:), 'bo',  'linewidth', 3, 'markersize', 8);
plot(mu(1,:), mu(2,:), 'rx-',  'linewidth', 3, 'markersize', 8);
for t=1:n, plot2dgauss(mu(1:2,t), V(1:2,1:2,t)); end
legend('observed', 'kalman filter')

figure;
hold on; grid on;
plot(X(1,:), X(2,:), 'bo',  'linewidth', 3, 'markersize', 8);
plot(nu(1,:), nu(2,:), 'rx-',  'linewidth', 3, 'markersize', 8);
for t=1:n, plot2dgauss(nu(1:2,t), U(1:2,1:2,t)); end
legend('observed', 'kalman smoother')
