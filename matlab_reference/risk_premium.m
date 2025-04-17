function res = risk_premium(gt,mt)
% This function is used to calculate the estimator of risk premium given
% SDF estimates and proxy factor gt

%% INPUT
% gt         is factor proxies (d by T)
% mt         is the estimator of SDF (1 by T)

%% OUTPUT
% rp         is d by 1 vector of risk premium

%% 
T = size(gt,2);
gtbar = gt - mean(gt,2);% 
mtbar = mt - mean(mt,2);% 
rp = -gtbar*mtbar'/T;

%% OUTPUT
res = rp;
  