function rank_loss = rankloss(ranks,Y)
%lets calculate rank error given ranks
%keyboard;
rank_loss = 0;
bigN = size(Y,1);
for i = 1:bigN
   %i
   rank_for_example = find(ranks(i,:)==Y(i));
   rank_loss = rank_loss + (1 - 1/rank_for_example); 
end
rank_loss = rank_loss/bigN;