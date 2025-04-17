function res = Ridge_rp(rt,lambda)

% This function returns the esitmator of SDF loading and SDF by Ridge regression with
% tuning parameter lambda

%% INPUT
% rt          is N by T matrix
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

mrt = mean(rt,2);
rtbar = rt - repmat(mrt,1,T);
Sigmahat = rtbar*rtbar'/T;

for j = 1:J
    b(:,j) = (Sigmahat+lambda(j)*eye(n))\mrt;
end

sdf                 =      1-b'*rtbar;

%% OUTPUT

res.sdf=sdf;
res.b = b;
    
  