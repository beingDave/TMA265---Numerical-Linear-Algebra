%% Clear All
clc; %clear all; close all;

%% Load Datasets: IRIS and Seals
[xi, yi, ci, xti, yti, cti, cl_x1i, cl_y1i, cl_x2i, cl_y2i] = load_iris;
[xs, ys, cs, hyps, cl_x1s, cl_y1s, cl_x2s, cl_y2s, sl, sw] = load_seals;

%% Set Parameters
plot_iris = 1;          % set to 1 to plot iris result
plot_seal = 1;          % set to 1 to plot seal results
plot_varylr = 1;        % set to 1 to plot results of varying learning rate on iris
lr = 0.5;               % learning rate
cntLt = 1e6;            % count limit for perceptron learning
% For Regularization => gamma_0 : 0.5 ; it passed directly when calling function

%% Compute on IRIS: Regularization Disabled
fprintf('---------------IRIS Dataset----------------\n');
fprintf('----------Regularization Disabled----------\n');
fprintf('-------------------------------------------\n');
[x1i, y1i, x2i, y2i, y3_1i, y3_2i, countsi, missClassNumi, missClassRatei] = compute_iris(xi, yi, ci, xti, yti, cti, lr, 0);

%% Compute on IRIS: Regularization Enabled
fprintf('---------------IRIS Dataset----------------\n');
fprintf('----------Regularization  Enabled----------\n');
fprintf('-------------------------------------------\n');
[x1ir, y1ir, x2ir, y2ir, y3_1ir, y3_2ir, countsir, missClassNumir, missClassRateir] = compute_iris(xi, yi, ci, xti, yti, cti, lr, 0.5);

%% Compute on Seals: Regularization Disabled
fprintf('---------------Seals Dataset---------------\n');
fprintf('----------Regularization Disabled----------\n');
fprintf('-------------------------------------------\n');
[y1s, x2s, y2s, y3_1s, y3_2s] = compute_seal(xs, ys, cs, hyps, 0, 0.5, cntLt, sl, sw);

%% Compute on Seals: Regularization Enabled
fprintf('---------------Seals Dataset---------------\n');
fprintf('----------Regularization  Enabled----------\n');
fprintf('-------------------------------------------\n');
[y1sr, x2sr, y2sr, y3_1sr, y3_2sr] = compute_seal(xs, ys, cs, hyps, 0.5, 0.5, cntLt, sl, sw);

%% Vary Learning Rate
fprintf('-------------------------------------------\n');
fprintf('---------- Varying Learning Rate ----------\n');
fprintf('-----------------On IRIS-------------------\n');

vary_lr = 0.1:0.1:0.8;
itr_l = zeros(size(vary_lr));
itr_q = itr_l; mc_l = itr_l; mc_q = itr_l;

for j = 1:length(vary_lr)
    [~, ~, ~, ~, ~, ~, lr_c, lr_num, ~] = compute_iris(xi, yi, ci, xti, yti, cti, vary_lr(j), 0);
    itr_l(j) = lr_c(1); itr_q(j) = lr_c(2);
    mc_l(j) = lr_num(1); mc_q(j) = lr_num(2);
end

%% Legends for Plotting
ls_legend = {'ls non-reg', 'ls reg', 'class 1', 'class 2'};
pll_legend = {'percentron linear non-reg', 'percentron linear reg', '          class 1', '          class 2'};
plq_legend = {'percentron quad non-reg', 'percentron quad reg', '          class 1', '          class 2'};

%% Plot Results
if plot_iris == 1
    pre = 'Iris DB: ';
    plot_comparison(1, x1i, y1i, x1ir, y1ir, cl_x1i, cl_x2i, cl_y1i, cl_y2i, ls_legend, sprintf('%sLeast Squares Comparison',pre), 'iris_');
    plot_comparison(2, x2i, y2i, x2ir, y2ir, cl_x1i, cl_x2i, cl_y1i, cl_y2i, pll_legend, sprintf('%sPerceptron Learning: Linear Comparison',pre), 'iris_');
    plot_comparison(3, sort(xi), y3_1i, sort(xi), y3_1ir, cl_x1i, cl_x2i, cl_y1i, cl_y2i, plq_legend, sprintf('%sPerceptron Learning: Quadratic Comparison',pre), 'iris_');

    plot_all(4, x1i, y1i, x2i, y2i, sort(xi), y3_1i, cl_x1i, cl_x2i, cl_y1i, cl_y2i, sprintf('%sRegularization Disabled',pre), 'iris_');
    plot_all(5, x1ir, y1ir, x2ir, y2ir, sort(xi), y3_1ir, cl_x1i, cl_x2i, cl_y1i, cl_y2i, sprintf('%sRegularization Enabled',pre), 'iris_');
end

if plot_seal == 1
    j = 1;
    if plot_iris == 1
        j = 5;
    end
    pre = 'Seals DB: ';
    plot_comparison(j+1, xs, y1s, xs, y1sr, cl_x1s, cl_x2s, cl_y1s, cl_y2s, ls_legend, sprintf('%sLeast Squares Comparison',pre), 'seal_');
    plot_comparison(j+2, x2s, y2s, x2sr, y2sr, cl_x1s, cl_x2s, cl_y1s, cl_y2s, pll_legend, sprintf('%sPerceptron Learning: Linear Comparison',pre), 'seal_');
    plot_comparison(j+3, sort(xs), y3_1s, sort(xs), y3_1sr, cl_x1s, cl_x2s, cl_y1s, cl_y2s, plq_legend, sprintf('%sPerceptron Learning: Quadratic Comparison',pre), 'seal_');

    plot_all(j+4, xs, y1s, x2s, y2s, sort(xs), y3_1s, cl_x1s, cl_x2s, cl_y1s, cl_y2s, sprintf('%sRegularization Disabled',pre), 'seal_');
    plot_all(j+5, xs, y1sr, x2sr, y2sr, sort(xs), y3_1sr, cl_x1s, cl_x2s, cl_y1s, cl_y2s, sprintf('%sRegularization Enabled',pre), 'seal_');
end

if plot_varylr == 1
    k = 11;
    figure(k);
    plot(vary_lr, mc_l, '-* r', 'linewidth', 1); hold on;
    plot(vary_lr, mc_q, '-o g', 'linewidth', 1); hold off;
    legend({'Order=1', 'Order=2'});
    title('Learning Rate vs Missclassications');
    xlabel('\eta');
    ylabel('Num. of Missclassifications');
    saveas(gcf, sprintf('vary_lr_lab30_%d',k), 'epsc');
    saveas(gcf, sprintf('vary_lr_lab30_%d',k), 'png');

    figure(k+1);
    plot(vary_lr, itr_l, 'r', 'linewidth', 1); hold on;
    plot(vary_lr, itr_q, 'g', 'linewidth', 1); hold off;
    title('Learning Rate vs Iteration Count');
    legend({'Order=1', 'Order=2'});
    xlabel('\eta');
    ylabel('Iterations');
    saveas(gcf, sprintf('vary_lr_lab30_%d',k+1), 'epsc');
    saveas(gcf, sprintf('vary_lr_lab30_%d',k+1), 'png');
end

%% Plotting Functions
function plot_comparison(i, x1, y1, x1_r, y1_r, cl_x1, cl_x2, cl_y1, cl_y2, l, t, pre)
    figure(i);
    plot(x1, y1, '-- r', 'linewidth', 1); hold on;
    plot(x1_r, y1_r, '-- g', 'linewidth', 1); hold on;
    plot(cl_x1, cl_y1, "ko", "MarkerSize", 5, "MarkerFaceColor", "c"); hold on;
    plot(cl_x2, cl_y2, "ks", "MarkerSize", 5, "MarkerFaceColor", "m"); hold off;
    title(t); legend(l); xlabel("x", "Interpreter", "Latex"); ylabel("y", "Interpreter", "Latex");
    saveas(gcf, sprintf('%s_lab30_%d',pre,i), 'epsc');
    saveas(gcf, sprintf('%s_lab30_%d',pre,i), 'png');
end

function plot_all(i, x1, y1, x2, y2, x3, y3, cl_x1, cl_x2, cl_y1, cl_y2, t, pre)
    figure(i);
    
    % decision line: least squares
    plot(x1, y1, '-- r', 'linewidth', 1); hold on;

    % decision line : perceptron "linear" learning algorithm
    plot(x2, y2, '-- g', 'linewidth', 1); hold on;

    % decision line : perceptron "quadratic" learning algorithm
    plot(x3, y3, '-- b', 'linewidth', 1); hold on;

    % data
    plot(cl_x1, cl_y1, "ko", "MarkerSize", 5, "MarkerFaceColor", "c"); hold on;
    plot(cl_x2, cl_y2, "ks", "MarkerSize", 5, "MarkerFaceColor", "m"); hold off;

    % plot details
    legend('least squares', 'perceptron : linear', 'perceptron : quadratic', '  class 1', '   class 2');
    xlabel("x", "Interpreter", "Latex");
    ylabel(" y ", "Interpreter", "Latex");
    title(t);
    saveas(gcf, sprintf('%s_lab30_%d',pre,i), 'epsc');
    saveas(gcf, sprintf('%s_lab30_%d',pre,i), 'png');
end