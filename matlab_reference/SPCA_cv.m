function res = SPCA_cv(param,return_variance)
% This function performs supervised PCA estimates of risk premium

%% INPUT

% rt          is n by T matrix
% gt          is d by T factor proxies
% usep        is # of factors we use (1 by J vector)
% tuning      is # of test assets we use at each step

%% OUTPUT

% Gammahat_nozero    is d by J vector of risk premia estimates
% b                  is J by N vector of SDF loading
% mimi               is d by N by J weights matrix for mimicking portfolio
% avarhat_nozero     is d by J vector of varaince estimator for rp
% alphahat           is d by J vector of pricing errors
% avarhat_alpha      is d by J vector of varaince estimator for alpha

%% INITIALIZATION

if ~isfield(param,'q')
   param.q = 1;
end

rt = param.rt;
gt = param.gt;
usep = param.usep;
N0 = param.tuning;
q = param.q;

T  =  size(rt,2);
n  =  size(rt,1);
d  =  size(gt,1);
J  =  length(usep);
pmax = max(usep);    
%% ESTIMATION
rtbar               =      rt - repmat(mean(rt,2),1,T);% 
gtbar               =      gt - repmat(mean(gt,2),1,T);% 
etahat_all          =      zeros(d,pmax);% is estimated eta
gammahat_all        =      zeros(pmax,1);% is estimated gamma
Index               =      [];% is the subset we choose at each step
k                   =      0;% is # of steps
B                   =      zeros(pmax,n);
mrt                 =      mean(rt,2);
vhat_all            =      zeros(pmax,T);

rt0                 =      rtbar;
gt0                 =      gtbar;
mrt0                =      mrt;
Gammahat_nozero     =      zeros(d,J);
b                   =      zeros(J,n);
mimi                =      zeros(d,n,J);
avarhat_nozero      =      zeros(d,J);
alphahat            =      zeros(d,J);
avarhat_alpha       =      zeros(d,J);
while(k<pmax)

    COR   = abs(corr(rt0',gt0'));

    L = max(COR,[],2);
    [bb,i] = sort(L);
    if N0<n
        II = (L >= bb(n-N0));
    else
        II = L>-1;
    end

    k = k + 1;

    Index(:,k) = 0;
    Index(II,k) = 1;

% perform PCA

    [U,S,V] = svds(rt0(II,:),1); 

    B(k,II) = U(:,1)'/S(1,1);
    gammahat_all(k,:) = U(:,1)' * mrt0(II)/S(1,1);
    etahat_all(:,k) = gt0*V(:,1);
    
% projection  
    gt0      =       gt0   -  gt0 * V(:,1) * V(:,1)'; 
    mrt0     =       mrt0    - rt0 * V(:,1) * gammahat_all(k,:);
    rt0      =       rt0   -  rt0 * V(:,1) * V(:,1)'; 
    vhat_all(k,:)  =       V(:,1)';
end


for jj = 1:length(usep)
    phat = usep(jj);
    vhat = vhat_all(1:phat,:);
    etahat = etahat_all(:,1:phat);
    gammahat = gammahat_all(1:phat,:);
    Sigmavhat             =       vhat*vhat'/T;
    what                  =       gtbar   -  etahat  *  vhat;

    if return_variance == true
% Newy-West Estimation
        Pi11hat               =        zeros(d*phat,d*phat);
        Pi12hat               =        zeros(d*phat,phat);
        Pi22hat               =        zeros(phat,phat);
        Pi13hat               =        zeros(d*phat,d);
        Pi33hat               =        zeros(d,d);

        for t = 1:T

            Pi11hat           =        Pi11hat  +  vec(what(:,t) * vhat(:,t)')*vec(what(:,t) * vhat(:,t)')'/T;
            Pi12hat           =        Pi12hat  +  vec(what(:,t) * vhat(:,t)')*vhat(:,t)'/T;
            Pi22hat           =        Pi22hat  +  vhat(:,t)     * vhat(:,t)'/T;

            Pi13hat           =        Pi13hat  +  vec(what(:,t) * vhat(:,t)')*what(:,t)'/T;
            Pi33hat           =        Pi33hat  +  what(:,t)     * what(:,t)'/T;

            for s = 1:min(t-1,q) 

                Pi11hat       =        Pi11hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vec(what(:,t-s) * vhat(:,t-s)')'+vec(what(:,t-s) * vhat(:,t-s)')*vec(what(:,t) * vhat(:,t)')');
                Pi12hat       =        Pi12hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vhat(:,t-s)' + vec(what(:,t-s) * vhat(:,t-s)')*vhat(:,t)');
                Pi22hat       =        Pi22hat  + 1/T*(1-s/(q+1))* (vhat(:,t)     * vhat(:,t-s)' + vhat(:,t-s) * vhat(:,t)' );

                Pi13hat       =        Pi13hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*what(:,t-s)' + vec(what(:,t-s) * vhat(:,t-s)')*what(:,t)');
                Pi33hat       =        Pi33hat  + 1/T*(1-s/(q+1))* (what(:,t)     * what(:,t-s)' + what(:,t-s) * what(:,t)' );

            end        
        end
        avarhat_nozero(:,jj) = diag(kron(gammahat'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*gammahat,eye(d))/T + ...
                     kron(gammahat'*inv(Sigmavhat),eye(d))*Pi12hat*etahat'/T + (kron(gammahat'*inv(Sigmavhat),eye(d))*Pi12hat*etahat')'/T + ...
                     etahat*Pi22hat*etahat'/T);
    end
    % risk premia estimator
    Gammahat_nozero(:,jj) = etahat * gammahat;
    
    B_sub = B(1:phat,:);
    fhat =  B_sub*rt;
    fhatbar = fhat - mean(fhat,2);
    mimi(:,:,jj) = (gt - mean(gt,2))*fhatbar'*(fhatbar*fhatbar')^(-1)*B_sub;
    b(jj,:) = mean(fhat,2)'*(fhatbar*fhatbar'/T)^(-1)*B_sub;

    % estimate alpha = E(g) - eta*gamma
    alphahat(:,jj) = mean(gt,2) - Gammahat_nozero(:,jj);
    if return_variance == true
        avarhat_alpha(:,jj) = diag(kron(gammahat'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*gammahat,eye(d))/T - ...
                         kron(gammahat'*inv(Sigmavhat),eye(d))*Pi13hat/T - (kron(gammahat'*inv(Sigmavhat),eye(d))*Pi13hat)'/T + ...
                         Pi33hat/T);
    end
end
%% Results
res.Gammahat_nozero = Gammahat_nozero;
res.b = b;
res.mimi = mimi;
res.alphahat = alphahat;
res.etahat = etahat_all;
res.gammahat = gammahat_all;
if return_variance == true
    res.avarhat_nozero = avarhat_nozero;
    res.avarhat_alpha = avarhat_alpha;
end
