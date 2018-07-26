close all;
clearvars;

% Import utility functions
addpath(genpath('../util'))

% Make results and visualization directories
mkdir('results');
mkdir('viz');

%% Experimental parameters

% Number of repetitions
nR = 1e4;

% Whether to visualize the problem setting
viz_setting = true;

% Whether to save figures
sav = false;

%% Problem setting

% Sample sizes
N = 2.^(2:1:5);
lN = length(N);

M = 2.^(2:1:5);
lM = length(M);

% Source parameters
mu_S = 0;
si_S = 0.75;

% Target parameters
mu_T = 0;
si_T = 1;

% 2D grid
nU = 501;
ul = [-10 +10];
u1 = linspace(ul(1),ul(2),nU);

%% Hypothesis and risk parameters

% Fixed parameter theta
th = 0.2;

% Analytical solution to target risk
RT = @(theta) theta.^2 - 2*theta./sqrt(pi) + 1;

%% Distribution functions

% Priors
pY = @(y) 1./2;

% Class-posteriors (pYS == pYT for covariate shift)
pYS = @(y,x) normcdf(y*x);
pYT = @(y,x) pYS(y,x);

% Source marginal distribution
pS = @(x) normpdf(x, mu_S, si_S);

% Source class-conditional distributions
pSY = @(y,x) (pYS(y,x) .* pS(x))./pY(y);

% Target marginal distribution
pT = @(x) normpdf(x, mu_T, si_T);

% Target class-conditional distributions
pTY = @(y,x) (pYT(y,x) .* pT(x))./pY(y);

% Importance weights
IW = @(x) pT(x) ./ pS(x);

% Helper functions for rejection sampling
pS_yn = @(x) pSY(-1,x);
pS_yp = @(x) pSY(+1,x);
pT_yn = @(x) pTY(-1,x);
pT_yp = @(x) pTY(+1,x);

%% Visualize setting

if viz_setting
    
    % Visualization parameters
    fS = 30;
    lW = 5;
    xx = linspace(-5,5,1001);
   
    % Initialize figure for source domain
    fg1 = figure(1);
    hold on
    
    % Plot source class-conditional distributions, p_S(x|y)
    plot(xx, pS_yn(xx), 'r', 'LineWidth', lW, 'DisplayName', '-1')
    plot(xx, pS_yp(xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');
    
    % Axes information
    legend('show')
    xlabel('$$x$$', 'Interpreter', 'latex');
    ylabel('$$p_{\cal S}(x|y)$$', 'Interpreter', 'latex');
    title(['Source domain, $$\sigma_S$$ = ' num2str(si_S)], 'Interpreter', 'latex');
    set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
    set(fg1, 'Color', 'w', 'Position', [0 0 1200 600]);
    
    if sav
        saveas(fg1, ['viz/source_dist_siS' num2str(si_S) '.png']);
    end
    
    % Initialize figure for target domain
    fg2 = figure(2);
    hold on
    
    % Plot target class-conditional distributions, p_S(x|y)
    plot(xx, pT_yn(xx), 'r', 'LineWidth', lW, 'DisplayName', '-1')
    plot(xx, pT_yp(xx), 'b', 'LineWidth', lW, 'DisplayName', '+1');
    
    % Axes information
    legend('show')
    xlabel('$$x$$', 'Interpreter', 'latex');
    ylabel('$$p_{\cal T}(x|y)$$', 'Interpreter', 'latex');
    title(['Target domain, $$\sigma_T$$ = ' num2str(si_T)], 'Interpreter', 'latex');
    set(gca, 'XLim', [-3 3], 'YLim', [0 1], 'FontSize', fS);
    set(fg2, 'Color', 'w', 'Position', [0 0 1200 600]);
    
    if sav
        saveas(fg2, ['viz/target_dist_siT' num2str(si_T) '.png']);
    end
end

%% Generate data

% Preallocate result arrays
RhS = zeros(lN,nR);
RhW = zeros(lN,nR);
RhT = zeros(lN,nR);

for r = 1:nR
    
    % Report progress over repetitions
    if (rem(r,nR./10)==1)
        fprintf('At repetition \t%i/%i\n', r, nR)
    end
    
    for n = 1:length(N)
        
        % Rejection sampling of target validation data
        const = 1.2./sqrt(2*pi*si_T);
        ss_Tn = round(M(n).*pY(-1));
        ss_Tp = M(n) - ss_Tn;
        Zy_n = sampleDist1D(pT_yn, const, ss_Tn, ul);
        Zy_p = sampleDist1D(pT_yp, const, ss_Tp, ul);
        
        % Rejection sampling of source data
        const = 1.2./sqrt(2*pi*si_S);
        ss_Sn = round(N(n).*pY(-1));
        ss_Sp = N(n) - ss_Sn;
        Xy_n = sampleDist1D(pS_yn, const, ss_Sn, ul);
        Xy_p = sampleDist1D(pS_yp, const, ss_Sp, ul);
        
        % Concatenate to datasets
        Z = [Zy_n; Zy_p];
        X = [Xy_n; Xy_p];
        u = [-ones(size(Zy_n,1),1); ones(size(Zy_p,1),1)];
        y = [-ones(size(Xy_n,1),1); ones(size(Xy_p,1),1)];
        
        % Importance weights
        W = IW(X);
        
        % Target risk of estimated theta
        RhS(n,r) = mean((X*th - y).^2, 1);
        RhW(n,r) = mean((X*th - y).^2 .* W, 1);
        RhT(n,r) = mean((Z*th - u).^2, 1);
        
    end
end

%% Boxplots of risks by estimator

% Visualization parameters
yT = 0.4:0.2:1.2;
fS = 15;
lW = 3;
labels = {'$$\hat{R}_{\cal S}$$',
          '$$\hat{R}_{\cal W}$$',
          '$$\hat{R}_{\cal T}$$'};
% Initialize figure
fg3 = figure(3);
hold on

for n = 1:length(N)
    subplot(1,lN,n)
    
    % Boxplot of estimated risk
    boxplot([RhS(n,:)', RhW(n,:)', RhT(n,:)'], 'Labels', labels);
    hold on;
    
    % Overlay average estimated risk
    plot([0.75, 1.25], [mean(RhS(n,:),2) mean(RhS(n,:),2)], 'k', 'LineWidth', lW);
    plot([1.75, 2.25], [mean(RhW(n,:),2) mean(RhW(n,:),2)], 'k', 'LineWidth', lW);
    plot([2.75, 3.25], [mean(RhT(n,:),2) mean(RhT(n,:),2)], 'k', 'LineWidth', lW);
    
    % Show true target risk
    line(1:3,[RT(th), RT(th), RT(th)], 'LineStyle', '--')
    
    % Axes options
    set(gca, 'YLim', [yT(1), yT(end)], 'FontSize', fS, 'YTick', yT, 'TickLabelInterpreter', 'latex');
    title(['N = M = ' num2str(N(n))], 'FontSize', fS, 'Interpreter', 'latex');
    
end

% Figure options
set(fg3, 'Color', 'w', 'Position', [10 100 1600 400]);

if sav
    saveas(fg2, ['viz\Rh_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

%% Histograms of risk by estimator

% Visualization parameters
yl = [0 0.15];
bins = linspace(prctile(RhW(:),5), prctile(RhW(:),95), 24);

fg4 = figure(4);
hold on

for n = 1:length(N)
    subplot(1,lN,n)

    % Plot normalized histograms
    histogram(RhW(n,:), bins, 'Normalization', 'Probability');
    
    % Axes information
    xlabel('$$\hat{R}_{\cal W}$$', 'Interpreter', 'latex')
    set(gca, 'YLim', [yl(1), yl(end)], 'XLim', [bins(1)-0.05 bins(end)+0.05], 'FontSize', fS);
    title(['N = ' num2str(N(n))]);
    
end

% Set figure information
set(fg4, 'Color', 'w', 'Position', [10 100 1600 400]);

if sav
    saveas(fg4, ['viz\hist_RhW_siS', num2str(si_S), '_siT' num2str(si_T) '_nR' num2str(nR) '.png'])
end

