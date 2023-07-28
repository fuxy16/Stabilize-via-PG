function [grad] = grad_eva(A,B,Q,R,r,K,gamma,n,m)
%  policy gradient by two-point gradient estimation
    M = 10*n; %100
    dif = 0;
    for j = 1:M
        U = randn(m,n);
        U = U./norm(U);
        K1 = K + r*U;
        K2 = K - r*U;
        x0 = randn(n,1);
        %x0=sqrt(3)*rand(n,1);
        V1 = oracle1(K1,Q,R,A,B,n,m,gamma,x0);
        V2 = oracle1(K2,Q,R,A,B,n,m,gamma,x0);
        dif = dif + (V1-V2)*U;
%         if isnan(dif)>0
%             2+2
%         end
    end
    grad = dif/(2*r*M);
    if max(grad(:))>100
        grad=100*grad/max(grad(:));
    end
end