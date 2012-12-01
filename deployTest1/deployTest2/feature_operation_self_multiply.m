function new_feature_matrix = feature_operation_self_multiply(matrix_a)
% takes one NxK matrices, and returns a N(N-1)/2 X K matrix which has the
% interactions between the features in them
%% self multiply 
 %the thing we would like to selfmultiply
k = size(matrix_a,2); %number of features
output = matrix_a; %first thing is the two original matrices themselves
for i = 1:k
    for j = i:k
        output = horzcat(output, matrix_a(:,i).*matrix_a(:,j)); %including every feature with itself: square terms
    end
end

new_feature_matrix = output;