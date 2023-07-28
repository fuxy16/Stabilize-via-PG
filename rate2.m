function [alpha] = rate2(Q, R, K, cost)
%return the update rule of the discount factor (for data-driven)
up = min(abs(eig(Q+K'*R*K)));
down = 2*cost-up;
alpha = (up/down);
end
