function ranks = make_final_prediction(model, example)
% Uses your trained model to make a final prediction for a SINGLE example.
%
% Usage:
%
%   RANKS = MAKE_FINAL_PREDICTION(MODEL, EXAMPLE);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes your trained model (output from INIT_MODEL) and a SINGLE 
% example, and returns a ranking ROW VECTOR as explained in the project
% overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Your model should be loaded from disk in INIT_MODEL. DO NOT DO
% ANY TRAINING HERE.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
%%
% We only take in one example at a time.
load('mfp_svm1.mat');
X_lyr = make_lyrics_sparse(example, model.vocab);
X_aud = make_audio(example);

% load model ==> model_lyr.model, model_aud.model
%% rescale
Xe_audio_scaled = rescalify(X_aud, model.modelaud.min, model.modelaud.range);
Xe_lyrics_scaled = rescalify(X_lyr, model.modellyr.min, model.modellyr.range);
Ye = zeros(size(X_aud,1),1); %just a dummy vector because svmpredict needs that
%clear X_aud;
%clear X_lyr;
%% Lyrics and Audio prediction based on the models
[Ye_pred_lyr, ~, Ye_pred_prob_estimates_lyr] = svmpredict(Ye, Xe_lyrics_scaled, model.modellyr.model, '-b 1');
[Ye_pred_aud, ~, Ye_pred_prob_estimates_aud] = svmpredict(Ye, Xe_audio_scaled, model.modelaud.model, '-b 1');

%% SINGLE CLASS OUTPUT
%combining the single-class predictions
Ye_pred_both_interact_binary = feature_operation_joined_binary_with_interaction(Ye_pred_aud, Ye_pred_lyr);

%%
%load level2 model
%load imputation order
[Ye_pred2, ~, Ye_pred2_prob_estimates] = svmpredict(Ye, Ye_pred_both_interact_binary, model.modell2);
ranks = impute(Ye_pred2, model.imputation_order_l2);
%{
% Find nearest neighbor
D = model.Xt*X';
[~,nn] = max(D);
yhat = model.Yt(nn);

% Convert into score vector
scores = zeros(1,10);
scores(yhat) = 1;

% Convert into ranks
ranks = get_ranks(scores);
%}
