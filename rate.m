function [alpha] = rate(Q, R, K, cost)
%return the update rule of the discount factor (for model-based)
up = min(eig(Q+K'*R*K));
down = cost;
alpha = (up/down);
end

