function res = FM(rt,gt)
% This function performs FM estimates of risk premium (two-pass procedure)

%%
rtbar = rt -mean(rt,2);
gtbar = gt -mean(gt,2);
betahat = (gtbar*gtbar')^(-1)*gtbar*rtbar'; 
gammahat = (betahat*betahat')^(-1)*betahat*mean(rt,2);
    
 %% Output   
 res.Gammahat_nozero = gammahat;
    
