rng default; % Setting the seed for reproducibility

model = garch(1,1); % Defining the GARCH(1,1) model

model.ARCH = {0.1}; % setting the arch lag1 
model.GARCH = {0.2}; % setting the garch lag1
model.Constant = .13; % setting the intercept for the model parameter

nSteps = 500;
N = 1;

[vol,log_returns] = simulate(model,nSteps,'NumPaths',N); % simulating the garch model 500 observations

figure
subplot(2,1,1)
plot(vol,'r')
xlim([0, nSteps])
title('Conditional Variances')
ylabel('Variance')

subplot(2,1,2)
plot(log_returns)
xlim([0, nSteps])
title('Innovations')
ylabel('innovations')

figure
autocorr(log_returns,100)
title('Log-Returns ACF')

figure
autocorr(log_returns.^2,100)
title('Square Log-Returns ACF')

figure
autocorr(abs(log_returns), 100)
title('ABS Log-Returns ACF')

% Testing stationarity 

[adf_h,adf_pValue, adf_stat, adf_cValue] = adftest(log_returns);

% Testing autocorrelation

[LB_h, LB_pValue, LB_stat, LB_cValue] = lbqtest(log_returns);

% Estimating parameters 

mld = garch(1,1); % define a new object with unknown parameters

[est_params,EstParamCov,logL,info] = estimate(mld, log_returns);
