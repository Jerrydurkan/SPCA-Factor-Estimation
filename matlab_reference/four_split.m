function res = four_split(rt,ft,p2,A1)
% This function performs Four-Split estimates of risk premium

%% INPUT
% rt          is n by T matrix
% ft          is pk by T factor
% p2          is # of potential missing factors
% A1          is p2 by pk matrix used to create instruments

%% OUTPUT

% Gammahat    is pk by 1 vector of risk premia estimator
% Sigmahat    is pk by pk matrix of covariance estimator

%% Estimation

T  =  size(rt,2);
n  =  size(rt,1);
pk = size(ft,1);
p = pk+p2;

%%

ftbar = ft-mean(ft,2);
rtbar = rt-mean(rt,2);
T1 = floor(T/4);
betahat = zeros(n,pk,4);

for j = 1:4
    a1 = T1*(j-1)+1;
    a2 = T1*j;
    ftbar1 = ftbar(:,a1:a2);
    rtbar1 = rtbar(:,a1:a2);
    betahat(:,:,j) = rtbar1*ftbar1'*(ftbar1*ftbar1')^(-1);
end
mrt = mean(rt,2);

period = [1 2 3 4; 2 3 4 1; 3 4 1 2; 4 1 2 3];
res1 = zeros(p,4);

X = zeros(n,p,4);
Z = zeros(n,2*pk,4);
subG = zeros(p,p,4);
tildeZ = zeros(p,n,4);
resid = zeros(n,1,4);
Ze = zeros(p,n,4);

for j=1:4
    J = period(j,:);
    X(:,:,j) = [betahat(:,:,J(1)), (betahat(:,:,J(1))-betahat(:,:,J(2)))*A1'];
    Z(:,:,j) = [betahat(:,:,J(3)), betahat(:,:,J(3))-betahat(:,:,J(4))];
    Xhat = Z(:,:,j)*(Z(:,:,j)'*Z(:,:,j))^(-1)*Z(:,:,j)'*X(:,:,j);
    tildeZ(:,:,j) = Xhat';
    subG(:,:,j) = X(:,:,j)'*Xhat/n;
    res1(:,j) = (Xhat'* Xhat)^(-1)* Xhat'*mrt;
    resid(:,:,j) = mrt- X(:,:,j)*res1(:,j);
    Ze(:,:,j) = tildeZ(:,:,j).*resid(:,:,j)';
end

R = kron(ones(4,1),[0.25*eye(pk);zeros(p2,pk)]);
G = blkdiag(subG(:,:,1),subG(:,:,2),subG(:,:,3),subG(:,:,4));
Ze_all = [Ze(:,:,1);Ze(:,:,2);Ze(:,:,3);Ze(:,:,4)];
Sigma0 = Ze_all*Ze_all'/n;
Sigmaf = ftbar*ftbar'/T;
Sigmahat = R'*G^(-1)*Sigma0*(G\R)/n+Sigmaf/T;

%% Results
res.Gammahat = mean(res1(1:pk,:),2);
res.Sigmahat = Sigmahat;