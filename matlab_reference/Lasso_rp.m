function res = Lasso_rp(rt,lambda)

% This function returns the esitmator of SDF loading and SDF by Lasso regression with
% tuning parameter lambda

%% INPUT
% rt          is returns (N by T matrix)
% lambda      is parameter for Ridge regression (J by 1 vector)

%% OUTPUT
% sdf     is Ridge SDF estimator for each lambda (J by T matrix)
% b       is estimator of SDF loading for each lambda (N by J matrix)

%% INITIALIZATION

T  =  size(rt,2);
n  =  size(rt,1);
J = length(lambda);
b = zeros(n,J);

%% ESTIMATION
c = norm(rt-mean(rt,2))/T^0.5/n^0.5;
rt = rt/c;
lambda = lambda/c;
mrt = mean(rt,2);
rtbar = rt - repmat(mrt,1,T);
Sigmahat = rtbar*rtbar'/T;

max_iter = 3000;
gamma = 1/norm(Sigmahat); %stepsize

for j = 1:J
    w = zeros(n,1);
    v = w;
    for t = 0:max_iter-1
        v_old = v;
        w_prev = w;
        w = v-gamma*(Sigmahat*v-mrt);
        w = sign(w).*max(abs(w)-lambda(j)*gamma,0);
        v = w+t/(t+3)*(w-w_prev);
        if (sum(power(v-v_old,2)) < (sum(power(v_old,2))*1e-5) || sum(abs(v-v_old))==0)
            break
        end
    end
    b(:,j) = v;
end
sdf  = 1-b'*rtbar;
%% OUTPUT

res.sdf=sdf;
res.b = b/c;