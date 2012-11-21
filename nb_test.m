function [y] = nb_test(nb, X)
% Generate predictions for a Gaussian Naive Bayes model.
%
% Usage:
%
%   [Y] = NB_TEST(NB, X)
%
% X is a N x P matrix of N examples with P features each, and NB is a struct
% from the training routine NB_TRAIN. Generates predictions for each of the
% N examples and returns a 0-1 N x 1 vector Y.
% 
% SEE ALSO
%   NB_TRAIN

% YOUR CODE GOES HERE (compute log_p_x_and_y)


%feature_i = 1
%y_value = 1 %either 1,2 for y = 0,1 => column in nb.mu_x_given_y
p_y_array = [1-nb.p_y, nb.p_y]; %[P(y=0), P(y=1)]
log_p_x_and_y = nan(size(X,1),2);
p_xi_yk = nan(size(X)); %number of features

for y_value = [1,2]
    for feature_i = 1:size(X,2) %number of features
        xi_minus_meani_given_yisy_value =  X(:,feature_i)-nb.mu_x_given_y(feature_i,y_value); %returns vector with one row per example
        p_xi_yk(:,feature_i) = 1/sqrt(2*pi*nb.sigma_x(feature_i).^2)*exp(-(xi_minus_meani_given_yisy_value).^2/(2*nb.sigma_x(feature_i).^2));
    end
    log_p_x_and_y(:,y_value) = sum(log(p_xi_yk),2) + log(p_y_array(y_value));
end

% Take the maximum of the log generative probability 
[~, y] = max(log_p_x_and_y, [], 2);
% Convert from 1,2 based indexing to the 0,1 labels
y = y -1;



    
