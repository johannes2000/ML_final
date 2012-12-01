function yhats = kernel_regression_test(Xtrain,Ytrain,Xtest,sigma)

N = size(Xtrain,1);
M = size(Xtest,1);
K = numel(unique(Ytrain));

kernel_matrix = zeros(M,N);
Xtrain = Xtrain';
Xtest = Xtest';
const = -1/(2*sigma^2);

for i = 1 : M
  % compute X2_i minus all X_j
  b_a = bsxfun(@minus,Xtest(:,i),Xtrain);
  % square each difference
  b_a_square = b_a.^2;
  % sum the squared differences
  sum_squares = sum(b_a_square);
  % constant operations
  row_i = exp(const .* sum_squares);
  % save the values in the kernel matrix
  kernel_matrix(i,:) = row_i;
end

yhats = zeros(M,K);
for i = 1 : K
  yhats(:,i) = sum(kernel_matrix(:,Ytrain == i),2);
end

end