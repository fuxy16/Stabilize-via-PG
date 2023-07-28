% stabilize_via_PG 
% Model-based implementation
%% parameters
clear
clc
close all
Q = [1,0;0,1];
R = 2;
A=[4,3;3,1.5];
B=[2;2];
n=2; % x dim
m=1; % u dim

step = 0.001; % policy gradient step size
num = 150;    % iteration
gamma = zeros(num,1);
optimal_gamma=zeros(num,1);

%% initialize
gamma(1) = 0.001;
K=zeros(m,n);
optimal_gamma(1) = get_optimal_gamma(A,B,K);

%% iteration
for i = 1:num
    % one step PG
    [nabla_K] = gradient(gamma(i),K,Q,R,A,B,n);
    K = K - step*nabla_K;
    [P] = value(gamma(i), A, B, Q, R, n, K);
    cost = trace(P);
    
    % update gamma
    [alpha] = rate(Q, R, K, cost);
    if gamma(i)/(1-alpha) > 1
        gamma(i+1) = 1;
        optimal_gamma(i+1)=min(get_optimal_gamma(A,B,K),1);
        break
    else
        optimal_gamma(i+1)=min(get_optimal_gamma(A,B,K),1);
        gamma(i+1) = gamma(i)/(1-alpha);
    end
end

%% plot
figure;
plot(optimal_gamma(1:i+1),'--','LineWidth',1.5,'color',[65 105 225]/255)
hold on 
plot(gamma(1:i+1),'LineWidth',1.5,'color',[255 153 18]/255)
set(gca,'FontSize',14);
legend('optimal factor','adaptive factor','Location','northwest')
xlabel('Iteration','FontSize',14)
ylabel('$\gamma$','FontSize',16,'Interpreter','latex','rotation',0)
set(gcf,'unit','centimeters','position',[1,2,14,6])
grid on
