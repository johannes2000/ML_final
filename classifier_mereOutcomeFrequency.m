function ranks = classifier_mereOutcomeFrequency(Y)

outcomes_frequencies = zeros(1,10);
N = size(Y,1);
for i = 1:10
   flag = (Y==i);
   outcomes_frequencies(i) = sum(flag)/N;
end

ranks = zeros(size(Y,1),10);

%keyboard;
%Populate a ranks matrix
for i = 1:10    
    %find the index of the largest probablity in that nx10 post array of
    %probabilities
    [~,I] = max(outcomes_frequencies); %this returns columns
    %for all examples delete cases we just copied over
    %assign that feature to the i'th column in the rank matrix
    ranks(:,i) = I;
    %now subtract thsi from posty
    outcomes_frequencies(I)=0;
    %keyboard;
end
