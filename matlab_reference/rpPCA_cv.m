function res = rpPCA_cv(param)
% This function performs rpPCA estimates of risk premium (risk premium PCA)

%% INPUT
% rt          is n by T matrix
% gt          is d by T factor proxies
% pmax        is upper bound of the number of factors
% tuning      is the tuning parameter for rpPCA (>=-1)

%% OUTPUT
% Gammahat_nozero    is d by pmax vector matrix of risk premia estimates
% sdf                is pmax by T vector of SDF estimates
% b                  is pmax by N vector of SDF loading
% mimi               is d by N by pmax weights matrix for mimicking portfolio

%%
rt = param.rt;
gt = param.gt;
pmax = param.pmax;
tuning = param.tuning;

%% INITIALIZATION

T = size(rt,2);
n = size(rt,1);
d = size(gt,1);
Gammahat_nozero = zeros(d,pmax);
b = zeros(pmax,n);
mimi = zeros(d,n,pmax);

%% ESTIMATION
gtbar = gt - repmat(mean(gt,2),1,T);% 
mu2 = (1+tuning)^0.5-1;
R = rt*(eye(T)+mu2/T*ones(T));
[U,S,V] = svd(R,'econ'); 

for p =1:pmax
    vhat = U(:,1:p)'*rt;
    vhatbar = vhat - repmat(mean(vhat,2),1,T);
    gammahat = mean(vhat,2);
    etahat = gtbar * vhatbar'*(vhatbar*vhatbar')^(-1);
    Gammahat_nozero(:,p) = etahat*gammahat;
    Sigmav = vhatbar*vhatbar'/T;
    b(p,:) = gammahat'*(Sigmav)^(-1)*U(:,1:p)';
    mimi(:,:,p) = gtbar * vhatbar'*(vhatbar*vhatbar')^(-1)*U(:,1:p)';
end
%% OUTPUT
res.Gammahat_nozero = Gammahat_nozero;
res.b = b;
res.mimi=mimi;
res.sdf = 1- b*(rt-mean(rt,2));
    
    
  