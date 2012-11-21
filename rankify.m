function ranks_for_submission = rankify(ranks)
% turn a n x m matrix of (probabilities, some numbers) into a sorted rannk 
% matrix for subm (also n by m). E.g. if for the first example the 5th number is the
% biggest in ranks, then ranks_for_sumission(1,1) = 5

%NOTE: this does tie breaking merely by picking up the first index
n = size(ranks,1);
m = size(ranks,2);

ranks_for_submission = zeros(n,m); %this creates an Nx10 rank matrix\
posty = ranks; %posty gets stepwise depleted

for mm = 1:m    
    [~,I] = max(posty,[],2); %this returns a vector I of colum ids
    ranks_for_submission(:,mm) = I;
     %set the max of every example to 0 in posty, so that it wont be pulled out next time
    for nn= 1:n %could be vectorized, but shouldn't take too much time
       posty(nn,I(nn)) = 0; 
    end
    %keyboard;
end