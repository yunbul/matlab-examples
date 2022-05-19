clc
clear all
close all
% ############################################################### INPUTS
% hyper parameters
Tpercent = 10; % size of portion taken from train set as labeled samples
Vpercent = 10; % size of portion taken from valid set as labeled samples
k = 21; % num of centroids used in feature clustering
voters = 7; % (!odd!) num of voters taken into account for majority voting

if Tpercent>100 || Tpercent<1 || k<1 ||...
        k<voters || voters<1 || mod(voters,2)==0
    display('Input parameters error.')
    return
end
% ############################################################### LOAD
% import mutual info toolbox
addpath(strcat(pwd,'\mi')); % compiled with log(2.0) fix
tic
% load train data with last column as labels
exT = matfile('gisette_train.data.mat'); Train = exT.Train;
exT = matfile('gisette_train.labels.mat'); Train = [Train exT.TrainLabels];
% load valid data
exT = matfile('gisette_valid.data.mat'); Valid = exT.Valid;
exT = matfile('gisette_valid.labels.mat'); Valid = [Valid exT.ValidLabels];
display(strcat('Data read in >>',num2str(toc),' seconds...'));
% ############################################################### SUBDATA
% randomly selected train instances 
% which are assumed to be labeled samples
% with a percentage on train size
indices=randperm(size(Train,1)); % generate random indices
s = size(Train,1)*Tpercent/100; % size
LabeledTrain=Train(indices(1:s),:);
% normalize data ignoring label column
[LabeledTrain] = normalization(LabeledTrain);
display(strcat('Random labeled train data generated with size >>',...
                num2str(s),' and normalized ...'));
% display class distribution of labeled train data
% expected to be balanced for a better training
figure(1)
hold on
hist(categorical(LabeledTrain(:,size(LabeledTrain,2)),[-1 1],{'-1','1'}));
title('Labeled Train Class Distribution')
hold off
% create random labeled valid data
indices=randperm(size(Valid,1));
s = size(Valid,1)*Vpercent/100;
LabeledValid=Valid(indices(1:s),:); 
LabeledValidLabels=LabeledValid(:,size(LabeledValid,2));
display(strcat('Random labeled valid data generated with size >>',...
                num2str(s),' ...'));
figure(2)
hold on
hist(categorical(LabeledValid(:,size(LabeledValid,2)),[-1 1],{'-1','1'}));
title('Labeled Valid Class Distribution')
hold off
% ############################################################### KMEANS
% slice labels from data and transpose for kmeans
LabeledTrainLabels = LabeledTrain(:,size(LabeledTrain,2)); % get labels
LabeledTrain = LabeledTrain'; % transpose data for feature clustering
LabeledTrain(size(LabeledTrain,1),:) = []; % remove labels from last row
display('In progress: k-means on features...')
tic
% apply kmeans on features to find diverse subsets
[idx, c] = kmeans(LabeledTrain,k,'Display','iter');
% get clusters
Cs = cell(k,1);
for i=1:size(LabeledTrain,1)
    Cs(idx(i),1) = {horzcat(cell2mat(Cs(idx(i),1)) ,i)};
end
display(strcat(num2str(k),' number of feature subsets created in >>',...
                num2str(toc),' seconds...'));
display('In progress: generating cluster silhouettes ...');
tic
figure(3)
hold on
[silho, h] = silhouette(LabeledTrain,idx);
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Spreads'
ylabel 'Clusters '
hold off
display(strcat('Completed in >>',num2str(toc),' seconds...'));
% ############################################################### CLASSIFY
% re-transpose data for data subset generation 
% for given feature subsets
LabeledTrain = LabeledTrain';
% get label predictions for all feature subsets
% also calculate each subset's accuracy 
% on small portions of train and valid datasets
predictions = []; 
LblTrainAccuracies = [];
LblValidAccuracies = [];
tic
% for each feature subset
display('Calculating accuracies on labeled datasets:...')
for i=1:k
    display(strcat('--Feat-subset',num2str(i),':'))
    % get labeled data subset with given feature subset
    featureSubset = cell2mat(Cs(i,1));
    SampleTrain = LabeledTrain(:,featureSubset);
    SVMModel = fitcsvm(SampleTrain,LabeledTrainLabels,...
        'KernelFunction','linear','Standardize',false,'ClassNames',[-1,1]);
    
    % calculate accuracy for given feature subset
    subsetPredictions = [];
    correctlyClassfied = 0;
    for j=1:size(SampleTrain,1)
        [label,score] = predict(SVMModel,SampleTrain(j,:));
        if LabeledTrainLabels(j) == label
            correctlyClassfied = correctlyClassfied + 1;
        end
        subsetPredictions = [subsetPredictions; label];
    end
    accuracy = 100*correctlyClassfied/j; 
    display(strcat('----Train:',num2str(accuracy),'%'));
    LblTrainAccuracies = [LblTrainAccuracies accuracy];
    predictions = [predictions subsetPredictions];
    
    SampleValid = LabeledValid(:,featureSubset);
    correctlyClassfied = 0;
    for j=1:size(SampleValid,1)
        [label,score] = predict(SVMModel,SampleValid(j,:));
        if LabeledValidLabels(j) == label
            correctlyClassfied = correctlyClassfied + 1;
        end
    end
    accuracy = 100*correctlyClassfied/j; 
    display(strcat('----Valid:',num2str(accuracy),'%'));
    LblValidAccuracies = [LblValidAccuracies accuracy];
end
display(strcat('Completed in >>',num2str(toc),' seconds...'));
% plot each subsets accuracy
figure(4)
hold on
bar([LblTrainAccuracies' LblValidAccuracies'],'grouped')
legend(strcat('Train (size:',num2str(size(LabeledTrain,1)),')'),...
        strcat('Valid (size:',num2str(size(LabeledValid,1)),')'));
title('Feature Subset Accuracies on Labeled Data');
set(gca, 'XTickLabel',{1:k},'XTick',1:k);
ylabel 'Accuracy (%)'
xlabel 'Feature Subsets'
hold off
% ############################################################### MRMR
predictions = [1:k; predictions]; % add subset names to first row
tic
%voters = 5; % must be odd for 'majority = sum(voter labels)>0'
bestSubsets = mrmr_mid_d(predictions,[0;LabeledTrainLabels],voters);
display(strcat('mrmr for >>',num2str(voters),' voters completed in >>',...
                num2str(toc),' seconds...'));
% ############################################################### SUBDATA
display('In progress: overall train/valid datasets voter predictions...');
tic
% get overall train and valid predictions for best feature subsets
% with assuming that we only know the LabeledTrain SVM model
TrainPredictions = []; 
ValidPredictions = []; 
for i=1:voters
    tic
    featureSubset = cell2mat(Cs(bestSubsets(i),1));
    SVMModel = fitcsvm(LabeledTrain(:,featureSubset),LabeledTrainLabels,...
        'KernelFunction','linear','Standardize',false,'ClassNames',[-1,1]);
    SampleTrain = Train(:,featureSubset);
    subsetPredictions = [];
    for j=1:size(SampleTrain,1)
        [label,score] = predict(SVMModel,SampleTrain(j,:)); %cant have lbls
        subsetPredictions = [subsetPredictions; label];
    end
    TrainPredictions = [TrainPredictions subsetPredictions];
    
    SampleValid = Valid(:,featureSubset);
    subsetPredictions = [];
    for j=1:size(SampleValid,1)
        [label,score] = predict(SVMModel,SampleValid(j,:)); %cant have lbls
        subsetPredictions = [subsetPredictions; label];
    end
    ValidPredictions = [ValidPredictions subsetPredictions];
    display(strcat('Voter >>',num2str(i),' for subset >>',...
       num2str(bestSubsets(i)),'[size:',num2str(size(featureSubset,2)),...
       '] completed in >>',num2str(toc),' seconds...'));
end
display(strcat('Completed in >>',num2str(toc),' seconds...'));
% ############################################################### VOTERS
labels = [-1 1]; % used to plot voting habits of each voter
subsetNames = {};
for i=1:size(bestSubsets,2)
    subsetNames = [subsetNames num2str(bestSubsets(i))];
end
freq=zeros(length(labels),voters);
for i=1:length(labels)
    for j=1:voters
        freq(i,j) = sum(TrainPredictions(:,j)==labels(i));
    end
end
freq = freq./size(TrainPredictions,1); % get 0-1 scale
figure(5)
hold on
bar(freq','stack')
legend(num2str(labels(1)),num2str(labels(2)))
title('Voters Labeling Tendencies on Train Data');
set(gca, 'XTickLabel',subsetNames(1,:),...
    'XTick',1:length(bestSubsets));
ylabel 'Frequencies'
xlabel 'Feature Subsets'
hold off
for i=1:length(labels)
    for j=1:voters
        freq(i,j) = sum(ValidPredictions(:,j)==labels(i));
    end
end
freq = freq./size(ValidPredictions,1); % get 0-1 scale
figure(6)
hold on
bar(freq','stack')
legend(num2str(labels(1)),num2str(labels(2)))
title('Voters Labeling Tendencies on Valid Data');
set(gca, 'XTickLabel',subsetNames(1,:),...
    'XTick',1:length(bestSubsets));
ylabel 'Frequencies'
xlabel 'Feature Subsets'
hold off
% ############################################################### ALLDATA
display('Calculating overall train/valid accuracies with given voters...');
tic
correctlyClassfied = 0;
for i=1:size(TrainPredictions,1)
    actual = Train(i,size(Train,2));
    % get majority of voters
    if sum(TrainPredictions(i,:)) > 0
        % odd num of voters classifies as 1
        if actual == 1
            correctlyClassfied = correctlyClassfied + 1;
        end
    else
        % odd num of voters classifies as -1
        if actual == -1
            correctlyClassfied = correctlyClassfied + 1;
        end
    end
end
TrainAccuracy = 100*correctlyClassfied/i;
display(strcat('Train: ',num2str(TrainAccuracy),' % in >>',...
                num2str(toc),' seconds...'));
tic
correctlyClassfied = 0;
for i=1:size(ValidPredictions,1)
    actual = Valid(i,size(Valid,2));
    % get majority of voters
    if sum(ValidPredictions(i,:)) > 0
        % odd num of voters classifies as 1
        if actual == 1
            correctlyClassfied = correctlyClassfied + 1;
        end
    else
        % odd num of voters classifies as -1
        if actual == -1
            correctlyClassfied = correctlyClassfied + 1;
        end
    end
end
ValidAccuracy = 100*correctlyClassfied/i;
display(strcat('Valid: ',num2str(ValidAccuracy),' % in >>',...
                num2str(toc),' seconds...'));
% ############################################################### ALLSUBS
% what if we apply all selected subsets (as merged) directly
display('Calculating overall train/valid accuracies with merged subsets...');
feats = [];
for i=1:voters
    feats = [feats cell2mat(Cs(bestSubsets(i),1))];
end
SVMModel = fitcsvm(LabeledTrain(:,feats),LabeledTrainLabels,...
        'KernelFunction','linear','Standardize',false,'ClassNames',[-1,1]);
SampleTrain = Train(:,feats);
correctlyClassfied = 0;
tic
for j=1:size(SampleTrain,1)
	[label,score] = predict(SVMModel,SampleTrain(j,:)); %cant have lbls
	if Train(j,size(Train,2)) == label
        correctlyClassfied = correctlyClassfied +1;
    end
end
TrainAccuracy = 100*correctlyClassfied/j;
display(strcat('Train: ',num2str(TrainAccuracy),' % in >>',...
                num2str(toc),' seconds...'));

SampleValid = Valid(:,feats);
correctlyClassfied = 0;
tic
for j=1:size(SampleValid,1)
	[label,score] = predict(SVMModel,SampleValid(j,:)); %cant have lbls
	if Valid(j,size(Valid,2)) == label
        correctlyClassfied = correctlyClassfied +1;
    end
end
ValidAccuracy = 100*correctlyClassfied/j;
display(strcat('Valid: ',num2str(ValidAccuracy),' % in >>',...
                num2str(toc),' seconds...'));
