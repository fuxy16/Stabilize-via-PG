% stabilize_via_PG 
% Data-driven implementation
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
r=0.02;% random size for PG

step = 0.001; % policy gradient step size
num = 250;    % iteration
ploted_num=10; % running times
gamma = zeros(num,ploted_num);
optimal_gamma=zeros(num,ploted_num);

%% iteration
for j=1:ploted_num
    % initialize
    K = zeros(1,2);
    gamma(1,j) = 0.001;
    optimal_gamma(1,j) = get_optimal_gamma(A,B,K);
    for i = 1:num
        % update gamma
        cost = oracle(K,Q,R,A,B,n,m,gamma(i,j));
        [alpha] = rate2(Q, R, K, cost);
        if gamma(i,j)/(1-alpha) > 1
            optimal_gamma(i+1,j)=min(get_optimal_gamma(A,B,K),1);
            gamma(i+1:num,j) = 1;
            optimal_gamma(i+1:num,j) = 1;
            break
        else
            optimal_gamma(i+1,j)=min(get_optimal_gamma(A,B,K),1);
            gamma(i+1,j) = gamma(i,j)/(1-alpha);
        end
        
        % one-step PG
        [nabla_K] = grad_eva(A,B,Q,R,r,K,gamma(i+1,j),n,m);
        K = K - step*nabla_K;
    end
end

%% plot
mean_gamma=mean(gamma,2);
var_gamma=sqrt(mean((gamma-mean_gamma).^2,2));

mean_optimal_gamma=mean(optimal_gamma,2);
var_optimal_gamma=sqrt(mean((optimal_gamma-mean_optimal_gamma).^2,2));

lower_gamma = mean_gamma-var_gamma;
upper_gamma = min(1,mean_gamma+var_gamma);
lower_optimal_gamma = mean_optimal_gamma-var_optimal_gamma;
upper_optimal_gamma = min(1,mean_optimal_gamma+var_optimal_gamma);

temp=1:num;
figure(2);
plot(mean_optimal_gamma,'--','LineWidth',1.5,'color',[65 105 225]/255)
hold on
plot(mean_gamma,'LineWidth',1.5,'color',[255 153 18]/255)
hold on
h = fill([temp fliplr(temp)], [lower_gamma', fliplr(upper_gamma')], [255 227 132]/255);
set(h,'edgealpha',0,'facealpha',0.7) 
hold on
h = fill([temp fliplr(temp)], [lower_optimal_gamma', fliplr(upper_optimal_gamma')], [176,224,230]/255);
set(h,'edgealpha',0,'facealpha',0.7) 
hold on
set(gcf,'unit','centimeters','position',[1,2,14,8])

set(gca,'FontSize',14);
xlabel('Iteration','FontSize',14)
ylabel('$\gamma$','FontSize',14,'Interpreter','latex','rotation',0)
legend('optimal discount factor','adaptive discount factor','Location','northwest')
