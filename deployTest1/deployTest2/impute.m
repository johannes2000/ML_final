function upgraded_preds = impute(y_pred, imputation_order)
%blows up a nx1 prediction to nxk, on the basis of a 1xk imputation_order
%vector
upgraded_preds = zeros(numel(y_pred),numel(imputation_order));
for n = 1:numel(y_pred)
    impy = imputation_order;
    impy(find(impy==y_pred(n)))=[];
    upgraded_preds(n,1) = y_pred(n);
    upgraded_preds(n,2:end) = impy;
end
    