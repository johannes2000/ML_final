function ranks = classifierP_NaiveBayes(Xtrain, Ytrain, Xquiz)
nb = NaiveBayes.fit(Xtrain, Ytrain);
%Ypred = nb.predict(nb, Xtest) %this is an Nx1 matrix
[post,Ypred] = posterior(nb,Xquiz);

%Ypred is the predicted class...first in rank-output
%post is a N-by-nb.nclasses matrix containing the posterior probability of
%each observation for each class. post(i,j) is the posterior probability of 
%point/example I belonging to class j. 

%keyboard;
ranks = zeros(size(Xquiz,1),10); %this creates an Nx10 rank matrix

%posty gets stepwise depleted
posty = post;

for i = 1:10    
    %find the index of the largest probablity in that nx10 post array of
    %probabilities
    [~,I] = max(posty,[],2); %this returns columns
    %for all examples delete cases we just copied over
    %assign that feature to the i'th column in the rank matrix
    ranks(:,i) = I;
    %now subtract thsi from posty
    for n= 1:size(Xquiz,1)
       posty(n,I(n)) = 0; 
    end
    %keyboard;
end
