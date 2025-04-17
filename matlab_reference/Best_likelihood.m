function res = Best_likelihood(rt,b,Sigma,mu)
% This function is used to select the best SDF estimator of Ridge and Lasso
% based on maximum likelihood given true covariance matrix, Sigma and mean,
% mu

%% INPUT
% rt          is n by T matrix
% b          is estimator of SDF loading for each lambda (N by J matrix)
% Sigma      is true covariance matrix (N by N)
% mu         is true mean vector (N by 1)

%% OUTPUT
% b_op          is N by 1 vector of best estimates of SDF loading
% sdf_op        is 1 by T vector of best SDF estimator
% sdf_all       is J by T vector of SDF estimators given J SDF loadings

%% INITIALIZATION
T  =  size(rt,2);
n  =  size(rt,1);
J  =  size(b,2);
r2 = zeros(J,1);
%% ESTIMATION
mrt = mean(rt,2);
rtbar = rt - repmat(mrt,1,T);

for j = 1:J
    r2(j) = -(mu-Sigma*b(:,j))'*(Sigma\(mu-Sigma*b(:,j)));
end

[argvalue, argmax] = max(r2);
sdf_all                 =      1-b'*rtbar;
sdf_op           =    sdf_all(argmax,:);

%% OUTPUT
res.b_op = b(:,argmax);
res.b_all = b;
res.sdf_op = sdf_op;
res.sdf_all = sdf_all;
res.op = argmax;
res.r2 = r2;

