function H = HankelMatrix(x,L)
% A function for constructing the Hankel matrix of size(nx*L,N-L+1) of a data
% sequence of size nx x N (columns are individual samples).
nx = size(x,1);
N = size(x,2);
H = nan(nx*L,N-L+1);

X = x(:);  % concatenate data to a single vector
for i = 1:N-L+1  % iterate over columns
    H(:,i) = X((i-1)*nx+1:(i-1)*nx+L*nx);
end