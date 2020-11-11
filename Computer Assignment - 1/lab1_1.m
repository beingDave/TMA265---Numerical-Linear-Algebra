%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computer Lab 1 : Code 2 
% Linearized Model : zT = c1*z + c2*T +c3*1
% c1 = T_0, c2 = log A, c3 = E-T_0*(log A)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fresh Start
clc; clf;
clear all;
close all;

%% Parameters
E_star = 6 * 1000;
A_star = exp(-2.64);
T_0_star = 400;

%% Vary Delta, Constant N
delta = 0:0.1:1;
len_d = length(delta);

ea = zeros(len_d,3);
ee = zeros(len_d,3);
et = zeros(len_d,3);

N = 125;

for j = 1:len_d
    [y, T] = genOriginalData(N, A_star, E_star, T_0_star);

    rng(3,'philox');
    alpha = randn(1, length(y));

    z = log(y + y.*(delta(j)*alpha));
    b = (z.*T)';
    A = [z' T' (T.^0)'];
    x = compute(A, b);

    for i=1:3                           % i = 1 : NE
        x_hat = x(:,i);                 % i = 2 : QR
        T_0_hat = x_hat(1);             % i = 3 : SVD
        A_hat = exp(x_hat(2));
        E_hat = x_hat(3) + (x_hat(1) * x_hat(2));

        ea(j,i) = relative_error(A_hat, A_star);
        ee(j,i) = relative_error(E_hat, E_star);
        et(j,i) = relative_error(T_0_hat, T_0_star);
    end
    %table(ea, ee, et)
end

errors = [ea; ee; et];
errornames = ["A vs Noise", "E vs Noise", "T0 vs Noise"];
xtitle = '{\delta} {\epsilon} [0, 1]';

for i=1:3
    e = errors(i:i+len_d-1, :);
    plotter(i, delta, e(:,1), e(:,2), e(:,3), xtitle,errornames(i), 'Northwest'); 
end

%% Vary N, Constant Delta
N = 50:25:250;
len_N = length(N);

ea = zeros(len_N,3);
ee = zeros(len_N,3);
et = zeros(len_N,3);

delta = 0.35;

for j = 1:len_N
    [y, T] = genOriginalData(N(j), A_star, E_star, T_0_star);

    rng(2,'philox');
    alpha = randn(1, length(y));

    z = log(y + y.*(delta*alpha));
    b = (z.*T)';
    A = [z' T' (T.^0)'];
    x = compute(A, b);

    for i=1:3                           % i = 1 : NE
        x_hat = x(:,i);                 % i = 2 : QR
        T_0_hat = x_hat(1);             % i = 3 : SVD
        A_hat = exp(x_hat(2));
        E_hat = x_hat(3) + (x_hat(1) * x_hat(2));

        ea(j,i) = relative_error(A_hat, A_star);
        ee(j,i) = relative_error(E_hat, E_star);
        et(j,i) = relative_error(T_0_hat, T_0_star);
    end
    %table(ea, ee, et)
end

errors = [ea; ee; et];
errornames = ["A vs N", "E vs N", "T0 vs N"];
xtitle = 'Number of Discretization Points';

for i=1:3
    e = errors(i:i+len_N-1, :);
    plotter(i+3, N, e(:,1), e(:,2), e(:,3), xtitle, errornames(i), 'Northeast'); 
end

fprintf('All Plots are saved to disk!\n');

%% Plot Everything
function p = plotter(i, x, y_ne, y_qr, y_svd, xtitle, ytitle, loc)
    h = figure(i);
    set(h,'visible','off')    
    p = plot(x, y_ne, x, y_qr, x, y_svd);
    
    p(1).Marker = '.';
    p(2).Marker = 'o';
    p(3).Marker = '*';
    
    p(1).MarkerSize = 12;
    p(2).MarkerSize = 15;
    p(3).MarkerSize = 13;
    
    p(1).Color = 'r';
    p(2).Color = 'g';
    p(3).Color = 'b';
    
    xticks('auto');
    xticklabels('auto');
    xlabel(xtitle);
    
    ylabel('Error Measure');
    yticks('auto');
    yticklabels('auto');
    
    title(sprintf('Relative Error in %s', ytitle));
    legend('Normal Equations', 'QR', 'SVD', 'Location', loc);
	saveas(gcf, 'lab11_'+strrep(ytitle, ' ', ''), 'epsc');
    saveas(gcf, 'lab11_'+strrep(ytitle, ' ', ''), 'png');
    clf;
end

%% Compute Function
function x = compute(A, b)
    x_ne = LLS_NE(A,b);
    x_qr = LLS_QR(A,b);
    x_svd = LLS_SVD(A,b);
    x = [x_ne x_qr x_svd];
end

%% Generate Data
function [y, T] = genOriginalData(N, A_star, E_star, T_0_star)
    T = 750:10:(750+(10*(N-1)));
    y = A_star * exp(E_star./(T-T_0_star));
end

%% Solution Method: Normal Equations
function x = LLS_NE(A,b)
    ATb = A'*b;
    ATA = A'*A;
    n = length(A(1,:));
    lowerChol = zeros(n);

    %Cholesky factorization
    for j = 1:1:n
        s1 = 0;
        for k = 1:1:j-1
            s1 = s1 + lowerChol(j,k)*lowerChol(j,k);
        end
        lowerChol(j,j) = (ATA(j,j)-s1)^(1/2);
        for i = j+1:1:n
            s2 = 0;
            for k = 1:1:j-1
                s2 = s2 + lowerChol(i,k)*lowerChol(j,k);
            end
            lowerChol(i,j) = (ATA(i,j)-s2)/lowerChol(j,j);
        end
    end

    % Solver for LL^T x = A^Tb:
    % Define z=L^Tx, then solve
    % Lz=A^T b to find z.
    % After by known z we get x.

    % forward substitution Lz=A^T b to obtain z

    for i = 1:1:n
        for k = 1:1:i-1
            ATb(i) = ATb(i) - ATb(k)*lowerChol(i,k);
        end
        ATb(i) = ATb(i)/lowerChol(i,i);
    end

    % Solution of L^Tx=z , backward substitution

    for i = n:-1:1
        for k = n:-1:i+1
            ATb(i) = ATb(i) - ATb(k)*lowerChol(k,i);
        end
        ATb(i) = ATb(i)/lowerChol(i,i);
    end

    % Obtained solution
    x = ATb;
end

%% Solution Method: QR Factorization
function x = LLS_QR(A,b)
    n = length(A(1,:));
    q = [];
    r = [];

    for i = 1:1:n
        q(:,i) = A(:,i);
        for j = 1:1:i-1
          r(j,i) = q(:,j)'*A(:,i);
          q(:,i) = q(:,i) - r(j,i)*q(:,j);
        end
        r(i,i) = norm(q(:,i));
        q(:,i) = q(:,i)/r(i,i);
    end

    % compute right hand side in the equation
    Rx = q'*b;

    % compute solution via backward substitution
    for i = n:-1:1
        for k = n:-1:i+1
            Rx(i)=Rx(i)-Rx(k)*r(i,k);
        end
        Rx(i) = Rx(i)/r(i,i);
    end

    x = Rx;
end

%% Solution Method: Singular Value Decomposition
function x = LLS_SVD(A,b)
    [U, S, V]=svd(A);

    UTb = U'*b;

    % choose tolerance
    tol = max(size(A))*eps(S(1,1));
    s = diag(S);
    n = length(A(1,:));

    % compute number of singular values > tol
    r = sum(s > tol);

    w = [(UTb(1:r)./s(1:r))' zeros(1,n-r)]';

    x = V*w;
end

%% Relative Error
function y = relative_error(x, x_star)
    y = abs(x-x_star)/abs(x_star);
end