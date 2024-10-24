function [U,V,ksi,delta,G,J,label] = DFRLICM_data(X,c,alpha,beta,K)
% Please cite the paper: Double Fuzzy Relaxlation Local Information C-Means Clustering
% Double Fuzzy Relaxlation Local Information C-Means Clustering for data clustering
% X:d*n,d is the dimension of a sample and n is the number of sample; c is the number of cluster.
% K is the number of neighbor sample, which can be tuned from 2 to 10 (or larger than 10).
% alpha and beta are two regularization parameters
% The value of alpha can be set within [0.002 0.01 0.05 0.5 1 10 100 500 1000 5000 10000 5e4 1e5];
% The value of beta can be set within [100 500 1000 5000 1e4 5e4 1e5 5e5 1e6 5e6 1e7 5e7 1e8 5e8 1e9 5e9];
% U:c*n, V:d*c
%% Initialization
[d,n] = size(X);
U = rand(c,n);
U = U./sum(U);
delta = (1/c)*ones(c,1);
ksi = zeros(d,n);
maxIter = 30; % The maxIter can be flexibly adjusted according to needs. 
oldV = zeros(d,c);
[~,nj] = find_nn(X',K);% nj：n*K; Finds K nearest neigbors for all datapoints in the dataset.
%% major cycle
for iter = 1:maxIter
    % Update V
    V = (X-ksi)*U'.^2./((sum(U'.^2,1))+eps);
    % Update ksi
    ksifenzi = V*(delta.*U.^2);
    ksifenmu = delta'*U.^2+eps;
    ksi = (1/(1+alpha))*(X-ksifenzi./ksifenmu);  
    % Update Gij
    G = zeros(c,n);
    for i = 1:c
        for j = 1:n
            Xk = X(:,nj(j,:));
            dkj = 1./(sqrt(L2_distance_1(Xk,X(:,j)))+1);
            uik = (1-U(i,nj(j,:))).^2;
            dki = L2_distance_1(Xk,V(:,i));
            G(i,j) =  sum(dkj.*uik'.*dki);
        end
    end
    % Update U
    Dji = L2_distance_1(X-ksi,V)+alpha*L2_distance_1(ksi,zeros(d,c));
    Ufenzi = (Dji'+G).*delta+eps;
    U = 1./Ufenzi./sum(1./Ufenzi,1);
    % Update delta
    S = sum(U.^2.*(Dji'+G),2);
    delta = EProjSimplex_new(-S/(2*beta)); % The method proposed in [34].
    % Compute the obj function value
    J(iter) = sum(delta.*S+beta*delta.^2);
    % Stopping condition
    newV = V;errorV = (newV-oldV).^2;errorV = sum(errorV(:));
    fprintf('Double Fuzzy Relaxlation Local Information C-Means: iteration count = %d, Loss = %f\n',iter,J(iter));
    if errorV <= 1e-5
        J(iter+1:end) = J(iter);
        break
    else
        oldV = V;
    end
end
[ ~ , label ] = max(U);
end