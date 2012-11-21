function [ypred ytest]= classifier_GeneralizedLinearModel(X, Y, Xtest)
mdl = GeneralizedLinearModel.stepwise(X,Y)
ypred = predict(mdl, X)
ytest = predict(mdl, Xtest)

%ytest = predict(mdl,Xtest)