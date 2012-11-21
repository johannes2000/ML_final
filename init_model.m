function [model] = init_model(vocab)
% Loads your precomputed trained model from disk.
%
% Usage:
%
%  MODEL = INIT_MODEL(VOCAB)
%
% VOCAB is the vocabulary we use at test time. It's the same as the one
% you were given, but if you didn't store it in your model, we give it to
% you now.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

load('my_model.mat','model');

