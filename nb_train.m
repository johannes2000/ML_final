function nb = nb_train(X, Y)
% Train a Gaussian Naive Bayes model with shared variances.
%
% Usage:
%
%   [NB] = NB_TRAIN(X, Y)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of 0-1 class labels. Returns a struct NB with fields:
%    nb.p_y          -- scalar, P(Y=1)
%    nb.mu_x_given_y -- P x 2 matrix of class means for each feature
%    nb.sigma_x      -- P x 1 matrix of standard deviations for each feature
% 
% SEE ALSO
%   NB_TEST

% **** NOTE: Variances should never be zero, even if the variance of the
% data is zero. Therefore you should always add a small positive constant
% to estimates of variance to prevent your prediction code from crashing.
% Use the matlab constant 'eps' for this.

%assuming that Y comes in {0,1}

nb.p_y = sum(Y==1)/numel(Y);% YOUR CODE GOES HERE

nb.mu_x_given_y(:,1) = sum(X(find(Y==0),:))/sum(Y==0);
nb.mu_x_given_y(:,2) = sum(X(find(Y==1),:))/sum(Y==1);

squared_deviations_given_yis0 = (bsxfun(@minus, X(find(Y==0),:), nb.mu_x_given_y(:,1)')).^2;
squared_deviations_given_yis1 = (bsxfun(@minus, X(find(Y==1),:), nb.mu_x_given_y(:,2)')).^2;

nb.sigma_x = sqrt( (sum(squared_deviations_given_yis0)+sum(squared_deviations_given_yis1))/numel(Y))'

%sum(X(find(Y==1),:))/sum(Y==1) is a 1x9 array of averages
%now subtract averages from the 


%{

size(bsxfun(@minus, X(find(Y==1),:), nb.mu_x_given_y(:,2)'))

ans =

   239     9

has the means subtracted for one class.

//

==> square it

size((bsxfun(@minus, X(find(Y==1),:), nb.mu_x_given_y(:,2)')).^2)

ans =

   239     9

//

%}