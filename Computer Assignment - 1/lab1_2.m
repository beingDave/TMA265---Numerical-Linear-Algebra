%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computer Lab 1 : Code 3
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

%% Vary N, Vary Delta
N = 100:25:225;
delta = 0.1:0.1:0.6;

len_N = length(N);
len_d = length(delta);

l = len_N*len_d;

ea = zeros(l,3);
ee = zeros(l,3);
et = zeros(l,3);

l = 1;

best_N = 200;
delta_range = [0 0.1 0.2 0.3];

for k = 1:len_N
    for j = 1:len_d
        [y, T] = genOriginalData(N(k), A_star, E_star, T_0_star);

        rng(2,'philox');
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
            
            ea(l,i) = relative_error(A_hat, A_star);
            ee(l,i) = relative_error(E_hat, E_star);
            et(l,i) = relative_error(T_0_hat, T_0_star);
        end
        l = l+1;
    end
end

plotter(delta, l-1, len_d, ea, ee, et, N);
check(best_N, delta_range, A_star, E_star, T_0_star);

%% Checker Function
function check(N, delta, A_star, E_star, T_0_star)
    [y, T] = genOriginalData(N, A_star, E_star, T_0_star);

    rng(2,'philox');
    alpha = randn(1, length(y));
    
    len_d = length(delta);
    
    t = zeros(len_d*3,3); k = 1;
    ea = zeros(len_d,3);
    ee = zeros(len_d,3);
    et = zeros(len_d,3);

    figure;

    for j = 1:len_d
        z = log(y + y.*(delta(j)*alpha));
        b = (z.*T)';
        A = [z' T' (T.^0)'];
        x = compute(A, b);
        
        for i=1:3                           % i = 1 : NE
            x_hat = x(:,i);                 % i = 2 : QR
            T_0_hat = x_hat(1);             % i = 3 : SVD
            A_hat = exp(x_hat(2));
            E_hat = x_hat(3) + (x_hat(1) * x_hat(2));
            y_hat = A_hat * exp(E_hat./(T-T_0_hat));
            t(k:k+2,i) = [A_hat, E_hat, T_0_hat];
            plot(y_hat,T);
            hold on;
            ea(j,i) = relative_error(A_hat, A_star);
            ee(j,i) = relative_error(E_hat, E_star);
            et(j,i) = relative_error(T_0_hat, T_0_star);
        end
        k=k+3;
    end
    fprintf('All Values:-\n'); table(t)
    fprintf('\n[Rounded 7 Digits] Relative Error in A:-\n'); table(round(ea,7))
    fprintf('\n[Rounded 7 Digits] Relative Error in E:-\n'); table(round(ee,7))
    fprintf('\n[Rounded 7 Digits] Relative Error in T0:-\n'); table(round(et,7))
    xlabel('function value (y)');
    ylabel('temperature (T)');
    legend(cellstr(num2str(delta', 'delta=%-.2f')));
    title(sprintf('y vs T for N=%d',N));
    hold off;
    saveas(gcf, 'lab12_check', 'epsc');
    saveas(gcf, 'lab12_check', 'png');
end

%% Plot Everything
function plotter(delta, l, n, ea, ee, et, N)
    figure('Position', [50 50 900 600]);
    for j=1:3
        for i = 1:n:l
            subplot(3,3,j)
            plot(delta, ea(i:i+n-1,j)); 
            xlabel('{\delta}'); ylabel('error in A'); xticks(delta); xticklabels('auto'); yticks('auto'); yticklabels('auto');
            if j == 1
                title("Normal Equation");
            elseif j == 2
                title("QR Factorization");
            elseif j == 3
                title('SVD');
            end
            hold on;
            subplot(3,3,j+3)
            plot(delta, ee(i:i+n-1,j)); 
            xlabel('{\delta}'); ylabel('error in E'); xticks(delta); xticklabels('auto'); yticks('auto'); yticklabels('auto');
            if j == 1
                title("Normal Equation");
            elseif j == 2
                title("QR Factorization");
            elseif j == 3
                title('SVD');
            end
            hold on;
            subplot(3,3,j+6)
            plot(delta, et(i:i+n-1,j));
            if j == 1
                title("Normal Equation");
            elseif j == 2
                title("QR Factorization");
            elseif j == 3
                title('SVD');
            end
            xlabel('{\delta}'); ylabel('error in T0'); xticks(delta); xticklabels('auto'); yticks('auto'); yticklabels('auto');
            hold on;
        end
    end
    legend(cellstr(num2str(N', 'N=%-d')), 'Orientation', 'horizontal', 'Location', 'none', 'Position', [0.5 0.035 0 0]);
    hold off;
    saveas(gcf, 'lab12_compare', 'epsc');
    saveas(gcf, 'lab12_compare', 'png');
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