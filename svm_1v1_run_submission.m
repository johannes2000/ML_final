%% load data

%clear;

load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);

Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);

%% partition Xt/Yt_lyrics
% into train and test set
% xtl = x_train_lyrics, ydl = y_development_lyrics
[xt_lyr xd_lyr yt_lyr yd_lyr] = partitionize(Xt_lyrics, Yt, 4);
%hist(yt_lyr,10);

%% Rescaling
data = xt_lyr;
xt_lyr_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
data = xd_lyr;
xd_lyr_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));

data = Xt_lyrics;
Xt_lyr_scaled = (data - repmat(min(data,[],1),size(data,1),1))*spdiags(1./(max(data,[],1)-min(data,[],1))',0,size(data,2),size(data,2));
%% training
tic
libsvmmodel = svmtrain(yt_lyr, xt_lyr_scaled,'-t 0 -b 1');
% t is kernel (0 is lin, 2 is RBF)
toc
tic
[yd_lyr_pred, accuracy, prob_estimates] = svmpredict(yd_lyr, xd_lyr, libsvmmodel, '-b 1');
toc

%%
tic
%using libSVM
libsvmmodel = svmtrain(y_temp, x_temp);
toc %1.4 seconds
tic
[yd_lyr_pred, accuracy, decision_values] = svmpredict(yd_lyr, xd_lyr, libsvmmodel);
toc
%training error
%ODL yt_lyr_pred = svmclassify(SVMStruct,xt_lyr);
TrainE = sum(yt_lyr_pred==yt_lyr)/size(yt_lyr,1);
%prediction on development set
%OLD yd_lyr_pred = svmclassify(SVMStruct,xd_lyr);
TestE = sum(yd_lyr_pred==yd_lyr)/size(yd_lyr,1);

%% OUTDATED IS UNDER LINE BELOW
 
%% make 1v1 classification matrix 
%(i.e. classification decision for every example from 10 different classifiers)
% relabel examples for each class: 1 if member of class, 0 if not
%xt_lyr 
%yt_lyr
class1_worked_on = 5;
class2_worked_on = 1;
class1_idx = find(yt_lyr == class1_worked_on);
class2_idx = find(yt_lyr == class2_worked_on);
%checked: [Yt(1:20), class1_idx(1:20)]
x_temp = xt_lyr([class1_idx; class2_idx],:); %pull the features for the 2 classes
y_temp = yt_lyr([class1_idx; class2_idx]);

%temp_y = yt_lyr; %ini
%1 v. all logic
%bench
%y_temp = (yt_lyr==class_worked_on); %checked
