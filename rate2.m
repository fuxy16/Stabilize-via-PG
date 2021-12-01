function [alpha] = rate(Q, R, K, cost)
%return the update rule of the discount factor (for data-driven)
up = min(eig(Q+K'*R*K));
down = 2*cost-up;
alpha = (up/down);
end
