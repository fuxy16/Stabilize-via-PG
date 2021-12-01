function [grad] = grad_eva(A,B,Q,R,r,K,gamma,n,m)
%  policy gradient by two-point gradient estimation
    M = 10;
    dif = 0;
    for j = 1:M
        U = 2*rand(m,n)-1;
        U = U./norm(U);
        K1 = K + r*U;
        K2 = K - r*U;
        x0 = randn(1)/sqrt(10);
        V1 = oracle1(K1,Q,R,A,B,n,m,gamma,x0);
        V2 = oracle1(K2,Q,R,A,B,n,m,gamma,x0);
        dif = dif + (V1-V2)*U;
    end
    grad = dif/(2*r*M);
end