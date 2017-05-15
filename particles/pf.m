function [z_pf, x_pf, w, time_pf] = pf(N,T,n_x,n_z,n_y,par,y,u,resamplingScheme)
% Particle Filter

% INITIALISATION:
% ==============
z_pf = ones(1,T,N);            % These are the particles for the estimate
                               % of z. Note that there's no need to store
                               % them for all t. We're only doing this to
                               % show you all the nice plots at the end.
z_pf_pred = ones(1,T,N);       % One-step-ahead predicted values of z.
x_pf = 10*randn(n_x,T,N);      % These are the particles for the estimate x.
x_pf_pred = x_pf;  
y_pred = 10*randn(n_y,T,N);    % One-step-ahead predicted values of y.
w = ones(T,N);                 % Importance weights.
initz = 1/n_z*ones(1,n_z);     
for i=1:N,
  z_pf(:,1,i) = length(find(cumsum(initz')<rand))+1; 
end;

tic;
for t=2:T,    
  fprintf('PF iter: %d\n',t);
  
  % SEQUENTIAL IMPORTANCE SAMPLING STEP:
  % =================================== 
  for i=1:N,
    % sample z(t)~p(z(t)|z(t-1))
    z_pf_pred(1,t,i) = length(find(cumsum(par.T(z_pf(1,t-1,i),:)')<rand))+1;
    % sample x(t)~p(x(t)|z(t|t-1),x(t-1))
    x_pf_pred(:,t,i) = par.A(:,:,z_pf_pred(1,t,i)) * x_pf(:,t-1,i) + ...
                       par.B(:,:,z_pf_pred(1,t,i))*randn(n_x,1) + ...
                       par.F(:,:,z_pf_pred(1,t,i))*u(:,t); 
  end;
  % Evaluate importance weights.
  for i=1:N,
    y_pred(:,t,i) =  par.C(:,:,z_pf_pred(1,t,i)) * x_pf_pred(:,t,i) + ...
                     par.G(:,:,z_pf_pred(1,t,i))*u(:,t); 
    Cov = par.D(:,:,z_pf_pred(1,t,i))*par.D(:,:,z_pf_pred(1,t,i))'; 
    w(t,i) =  (det(Cov)^(-0.5))*exp(-0.5*(y(:,t)-y_pred(:,t,i))'* ...
				    pinv(Cov)*(y(:,t)-y_pred(:,t,i))) + 1e-99;
  end;  
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
  z_pf(1,t,:) = z_pf_pred(1,t,outIndex);
  x_pf(:,t,:) = x_pf_pred(:,t,outIndex);

end

time_pf = toc; 

end