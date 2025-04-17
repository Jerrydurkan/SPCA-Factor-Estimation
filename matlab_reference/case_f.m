%% This code replicates simulation results in Giglio, Xiu, and Zhang (2021). If you use this code, please cite this paper. 
clear;
close all;
warning('off');

%% Calibration
load('Calibrated_factors.mat');

%%
l = 1; % task ID
%l = str2num(getenv('SLURM_ARRAY_TASK_ID'));

M = 1000;  % # of MCs
T = 120; % 180, 240   % # of time periods
n = 2000;   % # of stocks
p = 4;     % # of latent factors
d = 4;     % # of factor proxies
pmax = 6;     % the largest # of factors
q = 2;     % is # of lags used in Newy-West standard errors

a = 0.05;   % parameter for the strength of the weak factor


Sigmav = [Sigmav(1:3,1:3),zeros(3,1);zeros(1,3),5e-4];
beta0 = [1, zeros(1,p-1)];
Sigmau = 12e-4*eye(n);
Sigmaw = 5e-4*eye(d);
gamma = [gamma(1:3),0.004]';
xi = zeros(d,1);

rng(123); % Control the random seed
beta = repmat(beta0,n,1) + randn(n,p);
I = (1:n)>a*n;
e1 = randn(n,1);
e1(I,:) = e1(I,:)*0.1;    
beta(:,4) = -beta(:,3) + e1; % Correlated factor loadings
% beta(:,4) = e1;   % Weak factor loadings

eta = eye(4);
% eta = [1 0 0 0; 0 1 0 0; 0 0 0 1]; 

% True parameters
Sigma = beta*Sigmav*beta'+Sigmau;
mu = beta*gamma;
sdf_loading = Sigma\mu;

% Initialization

% Risk premium estimator
Gammatrue = eta*gamma;  

Gammahat_SPCA = zeros(d,M);
Gammahat_PCA = zeros(d,M);
Gammahat_PLS = zeros(d,M);
Gammahat_rpPCA = zeros(d,M);
Gammahat_Lasso = zeros(d,M);
Gammahat_Ridge = zeros(d,M);
Gammahat_four = zeros(d,M);
Gammahat_Fama = zeros(d,M);

% SDF estimator
sdf = zeros(T,M);
SDF_SPCA = zeros(T,M);
SDF_PCA = zeros(T,M);
SDF_rpPCA = zeros(T,M);
SDF_PLS = zeros(T,M);
SDF_Ridge = zeros(T,M);
SDF_Lasso = zeros(T,M);

% Sharpe ratio
sr_SPCA = zeros(1,M);
sr_PCA = zeros(1,M);
sr_rpPCA = zeros(1,M);
sr_PLS = zeros(1,M);
sr_Ridge = zeros(1,M);
sr_Lasso = zeros(1,M);

% Varaince estimator
avarhat_SPCA = zeros(d,M);
avarhat_PCA = zeros(d,M);

% Estimated # of factors
phat = zeros(1,M);

% Tuning parameters
tuningrange_SPCA = 100:100:500; % tuning range for SPCA (N0)
v_r = 10.^(-2:0.2:0); % tuning range for Ridge
v_l = 10.^(-2.4:0.2:-1.2); % tuning range for Lasso
mu_rp = -1:10; % tuning range for rpPCA


%% MC

tic
for iMC = 1:M
    rng((l-1)*M+iMC); % Control the random seed
    disp(iMC);
    
    vt = Sigmav^0.5*randn(p,T);          % factor innovations
    ut = Sigmau^0.5*randn(n,T);          % residual innovations
    wt = Sigmaw^0.5*randn(d,T);          % proxies residual innovations
    rt = repmat(beta*gamma,1,T) + beta*vt + ut; % returns
    gt = repmat(xi,1,T) + eta*vt + wt; % proxies
    
    sdf(:,iMC) = (1-sdf_loading'*(rt-mean(rt,2)))'; % true SDF
    
    
    %% Estimation    
    % SPCA
    param_spca.pmax = pmax; param_spca.rt = rt; param_spca.gt = gt;
    
    SPCAres = kfoldcv_tsr2(3,3,@SPCA_cv,param_spca,tuningrange_SPCA);
    
    Gammahat_SPCA(:,iMC) = SPCAres.Gammahat_nozero(:,end); % risk premia estimator
    b_SPCA = SPCAres.b(end,:); % estimated SDF loadings
    SDF_SPCA(:,iMC) = (1- b_SPCA*(rt-mean(rt,2)))'; % estimated SDF
    sr_SPCA(:,iMC) = ((b_SPCA*mu)'*(b_SPCA*Sigma*b_SPCA')^(-1)*(b_SPCA*mu))^0.5; % Sharpe Ratio
    phat(:,iMC) = SPCAres.pmax; % estimated # of factors
    
    %% PCA
    PCAres = three_pass_pca(rt,gt,p,q);
    
    Gammahat_PCA(:,iMC) = PCAres.Gammahat_nozero; % risk premia estimator
    SDF_PCA(:,iMC) = PCAres.sdf'; % estimated SDF
    sr_PCA(:,iMC) = ((PCAres.b*mu)'*(PCAres.b*Sigma*PCAres.b')^(-1)*(PCAres.b*mu))^0.5; % Sharpe Ratio
    
    %% Variance
    param2.rt = rt; param2.gt = gt; param2.q = q; param2.usep = SPCAres.pmax; param2.tuning = SPCAres.tuning;
    SPCAres_std = SPCA_cv(param2,true);
    
    avarhat_SPCA(:,iMC) = SPCAres_std.avarhat_nozero; % SPCA varaince estimator
    avarhat_PCA(:,iMC) = PCAres.avarhat_nozero; % PCA varaince estimator
    
    %% PLS
    PLSres = three_pass_pls(rt,gt,p);
    Gammahat_PLS(:,iMC) = PLSres.Gammahat_nozero;  % risk premia estimator
    SDF_PLS(:,iMC)   = PLSres.sdf'; % estimated SDF
    sr_PLS(:,iMC)   = ((PLSres.b*mu)'*(PLSres.b*Sigma*PLSres.b')^(-1)*(PLSres.b*mu))^0.5; % Sharpe Ratio
    
    %% rpPCA
    
    param_rp.pmax = p; param_rp.rt = rt; param_rp.gt = gt;
    sr_rpPCA_temp = zeros(length(mu_rp),1);
    
    for jjj = 1:length(mu_rp)
        param_rp.tuning = mu_rp(jjj);
        rpPCAres = rpPCA_cv(param_rp); 
        b_rpPCA = rpPCAres.b(end,:);
        sr_rpPCA_temp(jjj,1) = ((b_rpPCA*mu)'*(b_rpPCA*Sigma*b_rpPCA')^(-1)*(b_rpPCA*mu))^0.5; % calculate the SR for each tuning parameter
    end
    
    [sr_max,ind] = max(sr_rpPCA_temp); % Select the one with the largest OOS SR
    
    param_rp.tuning = mu_rp(ind);
    
    rpPCAres = rpPCA_cv(param_rp);

    Gammahat_rpPCA(:,iMC)   = rpPCAres.Gammahat_nozero(:,end); % risk premia estimator
    SDF_rpPCA(:,iMC)   = rpPCAres.sdf(end,:)'; % Estimated SDF
    b_rpPCA = rpPCAres.b(end,:); % Estimated SDF loadings
    sr_rpPCA(:,iMC)   = ((b_rpPCA*mu)'*( b_rpPCA*Sigma* b_rpPCA')^(-1)*( b_rpPCA*mu))^0.5; % Sharpe Ratio
    
    %% Ridge
    
    Ridgeres = Ridge_rp(rt,v_r);
    Best_r = Best_likelihood(rt,Ridgeres.b,Sigma,mu); % select the tuning with the largest likelihood (given true Sigma and mu)
    
    Gammahat_Ridge(:,iMC) = risk_premium(gt,Best_r.sdf_op);  % risk premia estimator
    SDF_Ridge(:,iMC) = Best_r.sdf_op'; % Estimated SDF
    b_Ridge = Best_r.b_op'; % Estimated SDF loadings
    sr_Ridge(:,iMC) = ((b_Ridge*mu)'*(b_Ridge*Sigma*b_Ridge')^(-1)*(b_Ridge*mu))^0.5; % Sharpe Ratio
    
    %% Lasso    

    Lassores = Lasso_rp(rt,v_l);
    Best_l = Best_likelihood(rt,Lassores.b,Sigma,mu); % select the tuning with the largest likelihood (given true Sigma and mu)
    
    Gammahat_Lasso(:,iMC)   = risk_premium(gt,Best_l.sdf_op); % risk premia estimator
    SDF_Lasso(:,iMC)   = Best_l.sdf_op';% Estimated SDF
    b_Lasso = Best_l.b_op'; % Estimated SDF loadings
    sr_Lasso(:,iMC)   = ((b_Lasso*mu)'*(b_Lasso*Sigma*b_Lasso')^(-1)*(b_Lasso*mu))^0.5; % Sharpe Ratio
    
     %% Four-split
    four_split_res = four_split(rt,gt,1,[1,0,0,0]);
    Gammahat_four(:,iMC) =  four_split_res.Gammahat;  % risk premia estimator
    
    %% Two-pass
    FMres = FM(rt,gt);
    Gammahat_Fama(:,iMC) = FMres.Gammahat_nozero;  % risk premia estimator
end
toc

%% Output
% Tables

% risk premia estimators
SPCAtab = [mean(Gammahat_SPCA - Gammatrue,2), mean((Gammahat_SPCA - Gammatrue).^2,2).^0.5];
PCAtab = [mean(Gammahat_PCA - Gammatrue,2), mean((Gammahat_PCA - Gammatrue).^2,2).^0.5];
rpPCAtab = [mean(Gammahat_rpPCA - Gammatrue,2), mean((Gammahat_rpPCA - Gammatrue).^2,2).^0.5];
Ridgetab = [mean(Gammahat_Ridge - Gammatrue,2), mean((Gammahat_Ridge - Gammatrue).^2,2).^0.5];
PLStab = [mean(Gammahat_PLS - Gammatrue,2), mean((Gammahat_PLS - Gammatrue).^2,2).^0.5];
Lassotab = [mean(Gammahat_Lasso - Gammatrue,2), mean((Gammahat_Lasso - Gammatrue).^2,2).^0.5];
fourtab = [mean(Gammahat_four - Gammatrue,2), mean((Gammahat_four - Gammatrue).^2,2).^0.5];
twotab =  [mean(Gammahat_Fama - Gammatrue,2), mean((Gammahat_Fama - Gammatrue).^2,2).^0.5];
result_rp = [Gammatrue,SPCAtab,PCAtab,rpPCAtab,PLStab,Lassotab,Ridgetab,fourtab,twotab];

% MSEs for SDF estimators
mse_SPCA = mean((SDF_SPCA-sdf).^2,1);
mse_PCA = mean((SDF_PCA-sdf).^2,1);
mse_Ridge = mean((SDF_Ridge-sdf).^2,1);
mse_Lasso = mean((SDF_Lasso-sdf).^2,1);
mse_rpPCA = mean((SDF_rpPCA-sdf).^2,1);
mse_PLS = mean((SDF_PLS-sdf).^2,1);

A=[mean(phat),mean(mse_SPCA),mean(mse_PCA),mean(mse_rpPCA),mean(mse_PLS),mean(mse_Lasso),mean(mse_Ridge)];
A2 = [std(phat),std(mse_SPCA),std(mse_PCA),std(mse_rpPCA),std(mse_PLS),std(mse_Lasso),std(mse_Ridge)];
result_mse = [A;A2];

% Sharpe Ratio for SDF estimators
B = [mean(sr_SPCA),mean(sr_PCA),mean(sr_rpPCA),mean(sr_PLS),mean(sr_Lasso,'omitnan'),mean(sr_Ridge)];
B2 = [std(sr_SPCA),std(sr_PCA),std(sr_rpPCA),std(sr_PLS),std(sr_Lasso,'omitnan'),std(sr_Ridge)];
C = (mu'*Sigma^(-1)*mu)^0.5;
result_sr = [B C;B2 0];


% CLT plots
nbin = 50; lw = 2;
set(gcf,'unit','normalized','position',[0.1,0.2,0.45,0.60]);
factor_name = ["RmRf","SMB","HML","V"];

for k = 1:d
    subplot(d,2,1+2*(k-1))
    binCtrs = linspace(-5,5,nbin);
    binWidth=binCtrs(2)-binCtrs(1);  
    counts=hist((Gammahat_SPCA(k,:) - Gammatrue(k))./sqrt(reshape(avarhat_SPCA(k,:),1,M)),binCtrs);
    prob = counts / (M * binWidth);   
    h1=bar(binCtrs,prob,'hist');

    set(h1,'FaceColor',[.6 .6 .6]);
    set(h1,'EdgeColor',[.6 .6 .6]);
            
    xgrid = linspace(-4,4,81);
    pdfReal = pdf('Normal',-4:0.1:4,0,1); 
    line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);    
    xlim([-5,5]);
    ylim([0,0.5]);
    title(['SPCA Result: ',char(factor_name(k))])
    
    subplot(d,2,2+2*(k-1));
    binCtrs = linspace(-5,5,nbin);
    binWidth=binCtrs(2)-binCtrs(1);  
    counts=hist((Gammahat_PCA(k,:) - Gammatrue(k))./sqrt(reshape(avarhat_PCA(k,:),1,M)),binCtrs);
    prob = counts / (M * binWidth);   
    h1=bar(binCtrs,prob,'hist');

    set(h1,'FaceColor',[.6 .6 .6]);
    set(h1,'EdgeColor',[.6 .6 .6]);
            
    xgrid = linspace(-4,4,81);
    pdfReal = pdf('Normal',-4:0.1:4,0,1); 
    line(xgrid,pdfReal,'color','k','linestyle','--','linewidth',lw);    
    xlim([-5,5]);
    ylim([0,0.5]);
    title(['PCA Result: ',char(factor_name(k))])
end

%% 
save('case_f.mat') 
