function [cost] = oracle1(K,Q,R,A,B, n, m, gamma, x0)
%% input policy and model parameters, output the cost
T = 100;
x = zeros(n, T);
u = zeros(m, T);
c = zeros(1,T);
x(:,1) = x0;
for k = 1:T-1
    u(:,k) = -K*x(:,k);
    %c(k) = gamma^(k-1)*(x(:,k)'*Q*x(:,k) + u(:,k)'*R*u(:,k));
    c(k) = x(:,k)'*Q*x(:,k) + u(:,k)'*R*u(:,k);
    x(:,k+1) = sqrt(gamma)*(A*x(:,k) + B*u(:,k));
end
cost = sum(c);
% if isnan(cost)>0
%     2+2
% end