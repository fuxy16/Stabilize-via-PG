function [cost] = oracle(K,Q,R,A,B, n, m, gamma)
%% input policy and model parameters, output the cost
T = 100;
num = 20;
cost = 0;

for j = 1:num
    x = zeros(n, T);
    u = zeros(m, T);
    c = zeros(1,T);
    x(:,1) = randn(n,1);
    for k = 1:T-1
        u(:,k) = -K*x(:,k);
        c(k) =(x(:,k)'*Q*x(:,k) + u(:,k)'*R*u(:,k));
        x(:,k+1) = sqrt(gamma)*(A*x(:,k) + B*u(:,k));
    end
    cost = cost + sum(c);
end
cost = cost/num;

