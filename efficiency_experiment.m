% efficiency of Algorithm
%% parameters
clear
clc
close all
Q = eye(100);
R = eye(100);
A=0.1*randn(100,100);
B=randn(100,100);

n=100; % x dim
m=100; % u dim
r=0.02;% random size for PG

step = 0.00001; % policy gradient step size
num = 2000;    % iteration
gamma = zeros(num,1);

%% initialize
gamma(1) = 0.001;
K=zeros(m,n);

%% iteration
for i = 1:num
    cost = oracle(K,Q,R,A,B,n,m,gamma(i));
    [alpha] = rate2(Q, R, K, cost);
    if gamma(i)/(1-alpha) > 1
        gamma(i+1:num) = 1;
        break
    else
        gamma(i+1) = gamma(i)/(1-alpha);
    end
    
    [nabla_K] = grad_eva(A,B,Q,R,r,K,gamma(i+1),n,m);
    K = K - step*nabla_K;
end
steps=i;
