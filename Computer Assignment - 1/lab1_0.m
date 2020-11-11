%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Computer Lab 1 : Code 1 
% Linearized Model : zT = c1*z + c2*T +c3*1
% c1 = T_0, c2 = log A, c3 = E-T_0*(log A)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Fresh Start
clc;
clear all;
close all;

%% Linearized Model Derivation
% log y = log(A) + E/(T - T_0)
% (log y) = (log A) + E/(T-T_0)
% (log y) - (log A) = E/(T-T_0)
% (log y)T - (log A)T - (log y)T_0 + (log A)T_0 = E
% (log y)T = + (log y)T_0 + (log A)T + E - (log A)T_0
% 
% c1 = T_0
% c2 = log A
% c3 = E-T_0*(log A)
% zT = c1*z + c2*T +c3*1

%% Generate Data
E_star = 6 * 1000;
A_star = exp(-2.64);
T_0_star = 400;
N = 20;
T = 750:10:(750+(10*(N-1)));
y = A_star * exp(E_star./(T-T_0_star));

%% Simple Linear Least Squares
% zT = c1*z + c2*T +c3*1
% c1=T_0, c2=log A, c3=E-T_0(log A)=E-c1*c2
z = log(y);
b = (z.*T)';
A = [z' T' (T.^0)'];
x_hat = A\b; % A*x = b => x = inv(A)*b

T_0_hat = x_hat(1);
A_hat = exp(x_hat(2));
E_hat = x_hat(3) + (x_hat(1) * x_hat(2));
y_hat = A_hat * exp(E_hat./(T-T_0_hat));

Error_A = relative_error(A_hat,A_star);
Error_E = relative_error(E_hat,E_star);
Error_T0 = relative_error(T_0_hat,T_0_star);
fprintf('Relative Error Table:\n');
table(Error_A,Error_E,Error_T0)

figure(1)
plot(y,T,'Marker','o','MarkerSize',11);
hold on;
plot(y_hat,T,'Marker','*','MarkerSize',10);
xlabel('function value (y)')
ylabel('temperature (T)')
legend('exact', 'computed')
title('y vs T')
saveas(gcf, 'lab10_a', 'epsc')
saveas(gcf, 'lab10_a', 'png')

function y = relative_error(x, x_star)
    y = abs(x-x_star)/abs(x_star);
end

% %% Using Optimization Solver 
% % zT = c1*z + c2*T +c3*1
% % c1=T_0, c2=log A, c3=E-T_0(log A)
% c = optimvar('c',3);
% func = c(1)*z + c(2)*T + c(3);
% obj = sum((z.*T - func).^2);
% lsqproblem = optimproblem("Objective",obj);
% 
% x0.c(1) = 100;
% x0.c(2) = log(1);
% x0.c(3) = 1000-(x0.c(1)*x0.c(2));
% [sol,fval] = solve(lsqproblem,x0)
% sol.c
% 
% T_0_hat_op = sol.c(1);
% A_hat_op = exp(sol.c(2));
% E_hat_op = sol.c(3) + (sol.c(1) * sol.c(2));
% y_hat_op = A_hat * exp(E_hat_op./(T-T_0_hat_op));
% 
% sprintf('Difference A_hat-A_star:\t\t%d\nDifference T_0_hat-T_0_star:\t%d\nDifference E_hat-E_star:\t\t%d', A_hat_op-A_star, T_0_hat_op-T_0_star, E_hat_op-E_star)
% 
% figure
% p=plot(y,T,y_hat,T, y_hat_op,T)
% p(1).LineWidth = 1;
% p(2).LineWidth = 1;
% p(3).LineWidth = 1;
% p(1).Marker = 'diamond';
% p(1).MarkerIndices = 1:2:length(T);
% p(2).Marker = '*';
% p(3).Marker = 'o';
% p(1).Color = 'r';
% p(2).Color = 'g';
% p(3).Color = 'b';
% legend('original', 'linearized ','linearized optimization prob')