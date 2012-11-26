function imputation_order = impute_make_order_based_on_mistakes(Ypred, Y)
%finds imputation order on the bases of wrong predictions in Ypred by mere
%frequency. Assumes 10 possible vals.
Ypred_wrong = find(Ypred~=Y);
edges = [.5:1:10.5];
[a,~] = histc(Y(Ypred_wrong),edges);
[~,imputation_order] = sort(a,'descend');
imputation_order = imputation_order(1:10); %artifact of edges histc

