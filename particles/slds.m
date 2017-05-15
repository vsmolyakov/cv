function [par, x, z, y, u] = slds(N,T,n_x,n_z,n_y)
%% init Switching Linear Dynamic System
par.A = zeros(n_x,n_x,n_z);
par.B = zeros(n_x,n_x,n_z);
par.C = zeros(n_y,n_x,n_z);
par.D = zeros(n_y,n_y,n_z);
par.E = zeros(n_x,n_x,n_z);
par.F = zeros(n_x,1,n_z);
par.G = zeros(n_y,1,n_z);
for i=1:n_z,
  par.A(:,:,i) = i*randn(n_x,n_x);
  par.C(:,:,i) = i*randn(n_y,n_x);
  par.B(:,:,i) = 0.01*eye(n_x,n_x);    
  par.D(:,:,i) = 0.01*eye(n_y,n_y);    
  par.F(:,:,i) = (1/n_x)*zeros(n_x,1);
  par.G(:,:,i) = (1/n_y)*zeros(n_y,1);   
end;

par.T = unidrnd(10,n_z,n_z);           % Transition matrix.
for i=1:n_z,
  par.T(i,:) = par.T(i,:)./sum(par.T(i,:)); 
end;

par.pz0 = unidrnd(10,n_z,1);            % Initial discrete distribution. 
par.pz0 = par.pz0./sum(par.pz0); 
par.mu0 = zeros(n_x,1);                 % Initial Gaussian mean.
par.S0  = 0.1*eye(n_x,n_x);             % Initial Gaussian covariance.  


%% generate data

x = zeros(n_x,T);         % unknown Gaussian states
z = zeros(1,T);           % discrete states
y = zeros(n_y,T);         % observations
u = zeros(1,T);           % control signal

x(:,1) = par.mu0 + sqrtm(par.S0)*randn(n_x,1);  %x0 ~ N(mu0,S0)
z(1) = length(find(cumsum(par.pz0')<rand))+1;   %z0 ~ P(z0)
for t=2:T,
  z(t) = length(find(cumsum(par.T(z(t-1),:)')<rand))+1;  %z_t ~ P(z_t|z_{t-1})
  x(:,t) = par.A(:,:,z(t))*x(:,t-1) + par.B(:,:,z(t))*randn(n_x,1) + par.F(:,:,z(t))*u(:,t); 
  y(:,t) = par.C(:,:,z(t))*x(:,t) + par.D(:,:,z(t))*randn(n_y,1) + par.G(:,:,z(t))*u(:,t); 
end;


end