% efficiency of Algorithm for n
%% parameters
clear
clc
close all

times=5;
ns=1:5;
n_max=length(ns);
trajectory_nums1=zeros(n_max,times);
trajectory_nums2=zeros(n_max,times);


for n=ns
    for t=1:times
        x_dim=2^(n); % x dim
        u_dim=4; % u dim
        
        Q = eye(x_dim);
        R = eye(u_dim);
        A=randn(x_dim,x_dim);
        A=2*(A+A')/max(abs(eig(A+A')));
        B=randn(x_dim,u_dim);
        B=B/max(abs(svd(B)));
        
        r=0.01;% random size for PG
        step = 0.001; % policy gradient step size 0.0001
        num = 100000;    % iteration num
        gamma = zeros(num,1);
        epsilon=0.7;
        optimal_P=dare(A,B,Q,R);
        %J_=1.01*trace(optimal_P);
        
        %% initialize
        gamma(1) = 1/(10*max(abs(eig(A)))^2);
        K=zeros(u_dim,x_dim);
        
        %% iteration
        trajectory_num1=0;
        trajectory_num2=0;
        for i = 1:num
            if i==num
                1000
            end
            cost = oracle(K,Q,R,A,B,x_dim,u_dim,gamma(i));
            trajectory_num1=trajectory_num1+20;
            [alpha] = rate2(Q, R, K, cost);
            if gamma(i)*(1+epsilon*alpha) > 1
                gamma(i+1:num) = 1;
                break
            else
                gamma(i+1) = real(gamma(i)*(1+epsilon*alpha));
            end
            
            %while  trace(value(gamma(i+1),A,B,Q,R,x_dim,K))>J_
                [nabla_K] = grad_eva(A,B,Q,R,r,K,gamma(i+1),x_dim,u_dim);
                K = K - step*nabla_K;
                trajectory_num2=trajectory_num2+2*10*(x_dim);
            %end
            
        end
        trajectory_nums1(n,t)=trajectory_nums1(n,t)+trajectory_num1;
        trajectory_nums2(n,t)=trajectory_nums2(n,t)+trajectory_num2;
    end
end

%% plot
close all
x=1:5;
y=log(mean(trajectory_nums2+trajectory_nums1,2))'/log(2);
a=polyfit(x,y,1);
plot(x,y,'b.','MarkerSize',25)
hold on
plot([0,6],[a(2),6*a(1)+a(2)])
set(gcf,'unit','centimeters','position',[1,2,10,8])
ax=gca;
ax.FontSize=14;
ax.XTick=0:6;
ax.XTickLabels={'2^0','2^1','2^2','2^3','2^4','2^5','2^6'};
%ax.YTick=[12,14,16,18,20,22,24];
%ax.YTickLabels={'2^{12}','2^{14}','2^{16}','2^{18}','2^{20}','2^{22}','2^{24}'};
xlabel('n','FontSize',14)
ylabel('Sample Complexity','FontSize',14)
grid on
