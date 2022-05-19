clc;
clear all;

[X] = load('train.txt');
[Y] = load('test.txt');

trainDataSize = size(X(:,1));
%testDataSize = size(Y(:,1));

X = vertcat(X,Y); % X is whole data now
                  % like train data followed by test vertically
                  % and ordered like 11..00../11..00.

% slice class labels from data
ClassLabels = X(:,[end]);
X = X(:,[1:end-1]);

% apply mean normalization on data
mu = mean(X);
X = bsxfun(@minus, X, mu);
sigma = std(X);
X = bsxfun(@rdivide, X, sigma);

%create eigenvectors U, eigenvalues S
[m, n] = size(X);
U = zeros(n);
S = zeros(n);
Sigma = X' * X / m;
[U, S, V] = svd(Sigma);

maxAccuracy = -1;
bestM = 0;
accuracyForMs = zeros(n,2);

%  Project the data onto m dimension
for m = n:-1:1
    tic;
    
    X_proj = zeros(size(X, 1), m);
    % projection of X into the reduced dimensional space 
    % spanned by the first m columns of U
    X_proj = X * U(:, 1:m);
    % add splitted class labels
    X_proj = horzcat(X_proj,ClassLabels);
    
    % split projected data into train and test again
    Y_proj = X_proj([trainDataSize+1 : end],:);
    X_proj = X_proj([1 : trainDataSize],:);
    
    %plot features if m=2
    if m==2
        figure; hold on; grid on;
        for c=1:size(X_proj(:,1))
            if X_proj(c,3) == 1
                plot(X_proj(c,1),X_proj(c,2),'x','Color', 'red');
            else
                plot(X_proj(c,1),X_proj(c,2),'+','Color', 'magenta');
            end
        end
        for c=1:size(Y_proj(:,1))
            if Y_proj(c,3) == 1
                plot(Y_proj(c,1),Y_proj(c,2),'x','Color', 'blue');
            else
                plot(Y_proj(c,1),Y_proj(c,2),'+','Color', 'cyan');
            end
        end
    end
    
    %KNN IMPLEMENTATION ######################################
    K = 5;
    [YpR YpC] = size(Y_proj);
    [XpR XpC] = size(X_proj);
    numFeats = YpC -1;
    classLblIndex = YpC;
    correctClassified = 0;
    
    for i=1:YpR %for all test samples
        
        neighs = zeros(XpR,2); %class-distance pairs
        for j=1:XpR %compare all train data
            dist = 0;
            trClassLbl = X_proj(j,classLblIndex);
            
            for featIndex=1:numFeats %find euclid sum for all features
                dist = dist + ((Y_proj(i,featIndex) - X_proj(j,featIndex)) * (Y_proj(i,featIndex) - X_proj(j,featIndex)));
            end
            
            dist = sqrt(dist); %take final sqrt
            neighs(j,1) = trClassLbl;
            neighs(j,2) = dist;
        end
        neighs = sortrows(neighs,2); % sort nearest neighbors
        
        topK = neighs([1:K],1); % get top K neighbor
        
        countOf1s = 0;countOf0s = 0;
        for k=1:size(topK(:,1))
            if topK(k,1) == 0
                countOf0s = countOf0s + 1;
            else
                countOf1s = countOf1s + 1;
            end
        end
        predict = 0;
        if countOf1s > countOf0s
            predict = 1;
        end
        if predict == Y_proj(i,classLblIndex)
            correctClassified = correctClassified + 1;
        end
        
    end
    %KNN IMPLEMENTATION ######################################
    accuracy = correctClassified*100/YpR;
    accuracyForMs(m,1) = m;
    accuracyForMs(m,2) = accuracy;
    if accuracy > maxAccuracy
        maxAccuracy = accuracy;
        bestM = m;
    end
    disp(strcat('Accuracy for m=',num2str(m),' , k=', num2str(K),' : ',num2str(accuracy),'% in :',num2str(toc),' sec.'));
    
end

accuracyForMs = sortrows(accuracyForMs,1);
figure; hold on; grid on; xlabel('m values'); ylabel('accuracy (%)');
plot(accuracyForMs(:,1),accuracyForMs(:,2),'-o','Color', 'red');

% 
% %recovered data
% X_rec = zeros(size(X_proj, 1), size(U, 1));
% X_rec = X_proj * U(:, 1:m)';

