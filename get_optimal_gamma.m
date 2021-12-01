function optimal_g=get_optimal_gamma(A,B,K)
%PLOT_OPTIMAL 此处显示有关此函数的摘要
%   此处显示详细说明
optimal_g=1/max(abs(eig(A-B*K)))^2;
end

