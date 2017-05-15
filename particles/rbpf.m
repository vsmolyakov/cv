function [z_rbpf, w, time_rbpf] = rbpf(N,T,n_x,n_z,n_y,par,y,u,resamplingScheme)
%Rao-Blackwellized Particle Filter

% INITIALISATION:
% ==============
z_rbpf = ones(1,T,N);          % These are the particles for the estimate
                               % of z. Note that there's no need to store
                               % them for all t. We're only doing this to
                               % show you all the nice plots at the end.
z_rbpf_pred = ones(1,T,N);     % One-step-ahead predicted values of z.
mu = 0.01*randn(n_x,T,N);      % Kalman mean of x.
mu_pred = 0.01*randn(n_x,N); 
Sigma = zeros(n_x,n_x,N);      % Kalman covariance of x.
Sigma_pred = zeros(n_x,n_x,N); 
S = zeros(n_y,n_y,N);          % Kalman predictive covariance.
y_pred = 0.01*randn(n_y,T,N);  % One-step-ahead predicted values of y.
w = ones(T,N);                 % Importance weights.
initz = 1/n_z*ones(1,n_z);     
for i=1:N,
  Sigma(:,:,i) = 1*eye(n_x,n_x); 
  Sigma_pred(:,:,i) = Sigma(:,:,i);
  z_rbpf(:,1,i) = length(find(cumsum(initz')<rand))+1; 
  S(:,:,i) = par.C(:,:,z_rbpf(1,1,i))*Sigma_pred(:,:,i)*par.C(:,:,z_rbpf(1,1,i))' + ...
                par.D(:,:,z_rbpf(1,1,i))*par.D(:,:,z_rbpf(1,1,i))';
end;


tic; 
for t=2:T,    
  fprintf('RBPF iter: %d\n',t);

  % SEQUENTIAL IMPORTANCE SAMPLING STEP:
  % =================================== 
  for i=1:N,
    % sample z(t)~p(z(t)|z(t-1))
    z_rbpf_pred(1,t,i) = length(find(cumsum(par.T(z_rbpf(1,t-1,i),:)')<rand))+1;
    
    % Kalman prediction:
    mu_pred(:,i) = par.A(:,:,z_rbpf_pred(1,t,i))*mu(:,t-1,i) + ... 
                   par.F(:,:,z_rbpf_pred(1,t,i))*u(:,t); 
    Sigma_pred(:,:,i)=par.A(:,:,z_rbpf_pred(1,t,i))*Sigma(:,:,i)*par.A(:,:,z_rbpf_pred(1,t,i))'...
                      + par.B(:,:,z_rbpf_pred(1,t,i))*par.B(:,:,z_rbpf_pred(1,t,i))'; 
    S(:,:,i)= par.C(:,:,z_rbpf_pred(1,t,i))*Sigma_pred(:,:,i)*par.C(:,:,z_rbpf_pred(1,t,i))' + ...
              par.D(:,:,z_rbpf_pred(1,t,i))*par.D(:,:,z_rbpf_pred(1,t,i))';  
    y_pred(:,t,i) = par.C(:,:,z_rbpf_pred(1,t,i))*mu_pred(:,i) + ... 
                      par.G(:,:,z_rbpf_pred(1,t,i))*u(:,t);
  end;
  % Evaluate importance weights.
  for i=1:N,
    w(t,i) = (det(S(:,:,i))^(-0.5))*  ...
             exp(-0.5*(y(:,t)-y_pred(:,t,i))'*pinv(S(:,:,i))*(y(:,t)- ...
						  y_pred(:,t,i))) + 1e-99; 
  end;  
%  w(t,:) = exp(log_w(t,:))+ 1e-99*ones(size(w(t,:)));
  w(t,:) = w(t,:)./sum(w(t,:));       % Normalise the weights.

  
  % SELECTION STEP:
  % ===============
  if resamplingScheme == 1
    outIndex = residualR(1:N,w(t,:)');        % Higuchi and Liu.
  elseif resamplingScheme == 2
    outIndex = deterministicR(1:N,w(t,:)');   % Kitagawa.
  else  
    outIndex = multinomialR(1:N,w(t,:)');     % Ripley, Gordon, etc.  
  end;
  z_rbpf(1,t,:) = z_rbpf_pred(1,t,outIndex);
  mu_pred = mu_pred(:,outIndex);
  Sigma_pred = Sigma_pred(:,:,outIndex);
  S = S(:,:,outIndex);
  y_pred(:,t,:) = y_pred(:,t,outIndex);


  % UPDATING STEP:
  % ==============
  for i=1:N,
    % Kalman update:
    K = Sigma_pred(:,:,i)*par.C(:,:,z_rbpf(1,t,i))'*pinv(S(:,:,i));
    mu(:,t,i) = mu_pred(:,i) + K*(y(:,t)-y_pred(:,t,i));
    Sigma(:,:,i) = Sigma_pred(:,:,i) - K*par.C(:,:,z_rbpf(1,t,i))*Sigma_pred(:,:,i);   
  end;

end

time_rbpf = toc;
end