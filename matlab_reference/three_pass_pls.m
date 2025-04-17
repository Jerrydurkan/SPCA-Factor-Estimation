function res = three_pass_pls(rt,gt,p)

% This function returns the esitmator of SDF loading and SDF by PLS given
% the number of factor p.

%% INPUT
% rt          is n by T matrix
% gt          is d by T factor proxies
% p           is number of PLS factors

%% OUTPUT
% Gammahat_nozero    is d by 1 vector matrix of risk premia estimates
% eta                is 1 by p vector of estimates
% gamma              is p by 1 vector of esimtates
% avarhat_nozero     is d by 1 vector of the avar of risk premia estimates
% vhat               is p by T vector of factor estimates
% sdf                is 1 by T vector of SDF estimates
% b                  is 1 by N vector of SDF loading

%% INITIALIZATION

T  =  size(rt,2);
n  =  size(rt,1);
d  =  size(gt,1);

%% ESTIMATION

rtbar = rt - repmat(mean(rt,2),1,T);
gtbar = gt - repmat(mean(gt,2),1,T);
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(rtbar',gtbar',p);
B = stats.W;
vhat = B'*rt;
vhatbar = vhat - repmat(mean(vhat,2),1,T);
gammahat = mean(vhat,2);
etahat = gtbar  *  vhatbar'*(vhatbar*vhatbar')^(-1);
Sigmav = vhatbar*vhatbar'/T;
b = gammahat'*(Sigmav)^(-1)*B';
Gammahat_nozero = etahat*gammahat;
sdf =  1 - b*rtbar;
%% OUTPUT

res.Gammahat_nozero = Gammahat_nozero;
res.eta = etahat*Sigmav^0.5;
res.gamma = Sigmav^(-0.5)*gammahat;
res.vhat = vhat;
res.b = b;
res.sdf = sdf;
res.B=B';