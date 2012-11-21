function ranks_output = voting(ranks_high, ranks_low, weight_for_high)
%% VOTING
%Weighted averaging between two nx10 submission-format rank matrices, based on the weight
%favoring the ranks_high. I.e. weight for high = 3, ranks_high votes will
%count 3 x as much as ranks_low

%Note: includes some logic to make it robust to such cases in which one
%matrix does not contain all 10 entries, e.g. has a row (9 9 5 4 2 ...)
%missing one number. In that case, the number that does not show up gets
%assigned rank 10. Not a good outcome, probably, throws warning.

ranks_output = zeros(size(ranks_high));
offset = 11; %not sure we would ever want to change that, uses a voting system such that 11 - position = score, e.g.
% position 2 => 11-2 = 9

%%for one example
for exy = 1:size(ranks_high,1)  %go through all examples, 1 by 1
    
    score_vec = zeros(10,1);
    for caty = 1:10 %% creates the scorign vector
        position_low = find(ranks_low(exy,:)==caty, 1); %%
        if isempty(position_low)
            position_low = 10;
            disp(['WARNING: ranks_low is missing a rank_order_entry ',num2str(caty),' in example ', num2str(exy)])
        end
        position_high = find(ranks_high(exy,:)==caty, 1);
        if isempty(position_high)
            position_high = 10;
            disp(['WARNING: ranks_high is missing a rank_order_entry ',num2str(caty),' in example ', num2str(exy)])
        end
        score_vec(caty) = (offset-position_low) + weight_for_high*(offset-position_high);
    end
    
    for i = 1:10 %% put together the rank order for that example based on the scoring
        %keyboard;
        [~,idx] = max(score_vec); %this returns the idx in the highest scoring vector
        ranks_output(exy,i) = idx; %index with the highest goes first
        score_vec(idx) = 0; %take this out for the next loop
    end
        %keyboard;
end
