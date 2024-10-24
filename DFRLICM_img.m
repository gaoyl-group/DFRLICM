function [U,V,ksi,delta,G,J,label] = DFRLICM_img(x,X,c,alpha,beta,K)
% Please cite the paper: Double Fuzzy Relaxlation Local Information C-Means Clustering
% Double Fuzzy Relaxlation Local Information C-Means for Image Segmentation
% x is the image for segmentation such as 200*200*3; X is the dataset reshaped by x
% X:d*n, d is the depth of the img. d of gray img is 1, color img is 3. n is the number of pixel.
% c is the number of cluster, alpha and beta are two regularization parameters
% The value of alpha can be set within [0.002 0.01 0.05 0.5 1 10 100 500 1000 5000 10000 5e4 1e5];
% The value of beta can be set within [1e7 4e7 8e7 1e8 4e8 8e8 1e9 4e9 8e9 1e10];
% K is the size of the local window;
% It is noted that the length of the side of the local window is 2*K+1.
% That is to say, when K = 1, the local window is 3*3
% U:c*n, V:d*c
%% Initialization
[d,n] = size(X);
if(d==1)
    [row ,col]=size(x); % n = row*col
else
    [row,col,~]=size(x);
end
U = rand(c,n);
U = U./sum(U);
delta = (1/c)*ones(c,1);
ksi = zeros(d,n);
maxIter = 200; % % The maxIter can be flexibly adjusted according to needs.
oldV = zeros(d,c);
%% major cycle
for iter = 1:maxIter
    % Update V
    V = (X-ksi)*U'.^2./((sum(U'.^2,1))+eps);
    % Update ksi
    ksifenzi = V*(delta.*U.^2);
    ksifenmu = delta'*U.^2+eps;
    ksi = (1/(1+alpha))*(X-ksifenzi./ksifenmu);  
    % Update Gij
    G=zeros(c,n);
    for i = 1:c
        for j = 1:n
            n_row = mod(j,row);
            if(n_row == 0)
                n_row = row;
            end
            n_col = ceil(j/row);
            L = n_col-K;
            R = n_col+K;
            Up = n_row-K;
            Down = n_row+K;
            if(L <= 0)
                L = 1;
            end
            if(Up <= 0)
                Up = 1;
            end
            if(R > col)
                R = col;
            end
            if(Down > row)
                Down = row;
            end 
            for i1 = Up:Down
                for j1 = L:R
                    if(~(i1==n_row&&j1==n_col))
                        G(i,j)=G(i,j)+(1/(sqrt((i1-n_row)^2+(j1-n_col)^2)+1))*(1-U(i,(j1-1)*row+i1))^2*(sum((double(X(:,(j1-1)*row+i1))-V(:,i)).^2));
                    end
                end
            end
        end
    end
    % Update U
    Dji = L2_distance_1(X-ksi,V)+alpha*L2_distance_1(ksi,zeros(d,c));
    Ufenzi = (Dji'+G).*delta+eps;
    U = 1./Ufenzi./sum(1./Ufenzi,1);
    % Update delta
    S = sum(U.^2.*(Dji'+G),2);% c*1
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