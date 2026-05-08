% 空间面板数据回归（一） with spatial/time fixed-effect tests
clear

% Read model data
A = readmatrix('ModelData.csv');

% Read compressed spatial weight matrix
gzFile = 'SpatialWeightMatrix.csv.gz';
csvFile = 'SpatialWeightMatrix.csv';

% Unzip only if the CSV file does not already exist
if ~isfile(csvFile)
    gunzip(gzFile);
end

W1_raw = readmatrix(csvFile);
% Time periods: 2010–2020
T = 11;

% Number of spatial units
N = size(A, 1) / T;

% Row-standardize spatial weight matrix
rowSum1 = sum(W1_raw, 2);
rowSum1(rowSum1 == 0) = 1;
W_1 = W1_raw ./ rowSum1;

% Dependent variable
y = A(:,5);

% Independent variables
x = A(:,[9,10,11,12]);

% ===================================================
% Spatial and time fixed-effect tests
% ===================================================

[nobs, K] = size(x);

% Spatial fixed effects model log-likelihood
fprintf('\nSpatial fixed effects model log-likelihood\n');

model = 1;
[ywith, xwith, meanny, meannx, meanty, meantx] = demean(y, x, N, T, model);

results_ols = ols(ywith, xwith);
sige = results_ols.sige * ((nobs - K) / nobs);

logliksfe = -nobs/2 * log(2*pi*sige) ...
            - 1/(2*sige) * results_ols.resid' * results_ols.resid;

disp(logliksfe)

% Time-period fixed effects model log-likelihood
fprintf('\nTime-period fixed effects model log-likelihood\n');

model = 2;
[ywith, xwith, meanny, meannx, meanty, meantx] = demean(y, x, N, T, model);

results_ols = ols(ywith, xwith);
sige = results_ols.sige * ((nobs - K) / nobs);

logliktfe = -nobs/2 * log(2*pi*sige) ...
            - 1/(2*sige) * results_ols.resid' * results_ols.resid;

disp(logliktfe)

% Two-way fixed effects model log-likelihood
fprintf('\nTwo-way fixed effects model log-likelihood\n');

model = 3;
[ywith, xwith, meanny, meannx, meanty, meantx] = demean(y, x, N, T, model);

results_ols = ols(ywith, xwith);
sige = results_ols.sige * ((nobs - K) / nobs);

loglikstfe = -nobs/2 * log(2*pi*sige) ...
             - 1/(2*sige) * results_ols.resid' * results_ols.resid;

disp(loglikstfe)

% Joint significance test: spatial fixed effects
LR = -2 * (logliktfe - loglikstfe);
dof = N;
probability = 1 - chis_prb(LR, dof);

fprintf('\nJoint significance of spatial fixed effects\n');
fprintf('LR = %9.4f, dof = %6d, p-value = %9.4f \n', ...
        LR, dof, probability);

% Joint significance test: time-period fixed effects
LR = -2 * (logliksfe - loglikstfe);
dof = T;
probability = 1 - chis_prb(LR, dof);

fprintf('\nJoint significance of time-period fixed effects\n');
fprintf('LR = %9.4f, dof = %6d, p-value = %9.4f \n', ...
        LR, dof, probability);

% ===================================================
% SAR panel fixed-effects model
% ===================================================

info.lflag = 1;
info.rmax = 1;
info.bc = 1;
info.model = 3;
info.fe = 0;

results = sar_panel_FE(y, x, W_1, T, info);

vnames = strvcat('SDG','GDP','Pop','Urb','Env');
prt_sp(results, vnames);