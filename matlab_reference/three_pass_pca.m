function res = three_pass_pca(rt,gt,p,q)
% This function performs PCA estimates of risk premium (three-pass procedure)

%% INPUT
% rt          is n by T matrix
% gt          is d by T factor proxies
% p            is the number of latent factors
% q           is # of lags used in Newy-West standard errors

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
rt_new              =      rt;
rtbar               =      rt_new - repmat(mean(rt_new,2),1,T);% 
gtbar               =      gt - repmat(mean(gt,2),1,T);% 

[U,S,V]             =      svd(rtbar,'econ'); 
gammahat            =      U(:,1:p)' * mean(rt,2)./diag(S(1:p,1:p));
etahat              =      gtbar  *  V(:,1:p);

vhat = V(:,1:p)';
Sigmavhat             =       vhat*vhat'/T;
what                  =       gtbar - gtbar * V(:,1:p) * V(:,1:p)';
phat                  =       p;

% Newy-West Estimation
Pi11hat               =        zeros(d*phat,d*phat);
Pi12hat               =        zeros(d*phat,phat);
Pi22hat               =        zeros(phat,phat);

for t = 1:T

    Pi11hat           =        Pi11hat  +  vec(what(:,t) * vhat(:,t)')*vec(what(:,t) * vhat(:,t)')'/T;
    Pi12hat           =        Pi12hat  +  vec(what(:,t) * vhat(:,t)')*vhat(:,t)'/T;
    Pi22hat           =        Pi22hat  +  vhat(:,t)     * vhat(:,t)'/T;

    for s = 1:min(t-1,q) 

        Pi11hat       =        Pi11hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vec(what(:,t-s) * vhat(:,t-s)')'+vec(what(:,t-s) * vhat(:,t-s)')*vec(what(:,t) * vhat(:,t)')');
        Pi12hat       =        Pi12hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vhat(:,t-s)' + vec(what(:,t-s) * vhat(:,t-s)')*vhat(:,t)');
        Pi22hat       =        Pi22hat  + 1/T*(1-s/(q+1))* (vhat(:,t)     * vhat(:,t-s)' + vhat(:,t-s) * vhat(:,t)' );

    end        
end
avarhat_nozero = diag(kron(gammahat'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*gammahat,eye(d))/T + ...
                 kron(gammahat'*inv(Sigmavhat),eye(d))*Pi12hat*etahat'/T + (kron(gammahat'*inv(Sigmavhat),eye(d))*Pi12hat*etahat')'/T + ...
                 etahat*Pi22hat*etahat'/T);
% Estimation of risk premium
Gammahat_nozero = etahat * gammahat;
% Estimation of loadings
sdf =  1-gammahat'*vhat*T;
B = U(:,1:p)';
fhat =  B*rt;
fhatbar = fhat-mean(fhat,2);
b = mean(fhat,2)'*(fhatbar*fhatbar'/T)^(-1)*B;

%% OUTPUT

res.Gammahat_nozero = Gammahat_nozero;
res.eta = etahat/T^0.5;
res.gamma = gammahat*T^0.5;
res.avarhat_nozero = avarhat_nozero;
res.vhat = T^0.5*vhat;
res.sdf = sdf;
res.b = b;
    
  