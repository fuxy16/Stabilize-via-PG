function [nabla_K] = gradient(gamma,K,Q,R,A,B,state_dim)
% policy gradient(model based)
% P iteration
I = eye(state_dim);
P0 = zeros(state_dim,state_dim);
iter = 100;
for o = 1:iter
    P1 = Q + K'*R*K + gamma*(A-B*K)'*P0*(A-B*K);
    P0 = P1;
end
P_K = P0;
% Sigma iteration
Sigma0 = zeros(state_dim,state_dim);
for o = 1:iter
    Sigma1 = I + gamma*(A-B*K)'*Sigma0*(A-B*K);
    Sigma0 = Sigma1;
end
Sigma_K = Sigma0;
E_K = (R+gamma*B'*P_K*B)*K - gamma*B'*P_K*A;
% gradient
nabla_K = 2*E_K*Sigma_K;

end

