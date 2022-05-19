clc
clear all
close all

% MIN~MAX ERROR RESULTS (approx) :
% 20 seq - 100 state >>> error = pi:[0.05~0.20], A:[0.005~0.03]
% 20 seq - 1000 state>>> error = pi:[0.05~0.20], A:[0.002~0.01]
% 100 seq - 100 state>>> error = pi:[0.01~0.10], A:[0.002~0.01]
% 100 seq - 1000 state>> error = pi:[0.01~0.07], A:[0.001~0.005]
%
% final verdict : if we want to increase prediction accuracy on 'pi' 
% we need to increase number of sequences, but further increments
% on 'A' requires more transitions to be applied on each sequence
% thus we need longer sequences, which makes a reasonable sense since
% 'pi' is estimated from the first column of sequences matrix 'Qs',
% whereas 'A' is derived by iterating along the each sequence itself.

%% PART I ################################################################
pi = [0.5, 0.2, 0.3];
A = [0.4 0.3 0.3; 0.2 0.6 0.2; 0.1 0.1 0.8];

Qs = []; % sequences
nQ = 100; % num of seq //20, 100
lQ = 1000;% length of each seq //100, 1000

% generate nQ number of sequences with length lQ
for i=1:nQ
    Qs = [Qs; GetSequence(pi, A, lQ)];
end
%% PART II ###############################################################
% initial probability estimate is the number of sequences starting with Si
% divided by the number of sequences.
nS = length(pi); % num of states

% estimate pi's from generated sequences
pi_estimate = zeros(1,nS);
for i=1:nS      % for each state
    for j=1:nQ          % for each sequence
        if Qs(j,1) == i % if starts with that state
            pi_estimate(i) = pi_estimate(i) + 1; % increase total num
        end
    end
end
pi_estimate = pi_estimate./nQ; % divide by num of seqs

% estimate for aij is the number of
% transitions from Si to Sj divided by the total number of transitions 
% from Si over all sequences.
A_estimate = zeros(nS);
for i=1:nS     %rows of A (from Si)
    for j=1:nS %cols of A (to Sj)
        trans_i = 0;   % num of transitions from i
        trans_i_j = 0; % num of transitions from i to j
        % for all sequences
        for k=1:nQ
            for n=1:lQ
                if Qs(k,n) == i % if transition from i
                    trans_i = trans_i + 1;
                    if Qs(k,n+1) == j % if transition from i to j
                        trans_i_j = trans_i_j + 1;
                    end
                end
            end
        end
        A_estimate(i,j) = trans_i_j / trans_i;
    end
end
% estimation errors
error_pi = abs(pi - pi_estimate); mae_pi = sum(error_pi)/3;
error_A = abs(A - A_estimate); mae_A = sum(sum(error_A,2))/9;

%% PART III ##############################################################
% read data.txt and convert A, B to states 1, 2
Os = zeros(500,20); %observations
fid = fopen('data.txt');
tline = fgetl(fid);
i = 1; %rows
while ischar(tline)
    for j=1:length(tline) %cols
        if tline(j) == 'A'
            Os(i,j) = 1;
        else
            Os(i,j) = 2;
        end
    end
    tline=fgetl(fid);
    i = i + 1;
end
fclose(fid);
Train = Os(1:300,:);
Valid = Os(301:500,:);

% same calculations in Part II
nS = 2; %num of states
[nQ, lQ] = size(Train);
pi_Train = zeros(1,nS);
for i=1:nS      
    for j=1:nQ  
        if Train(j,1) == i 
            pi_Train(i) = pi_Train(i) + 1; 
        end
    end
end
pi_Train = pi_Train./nQ;

A_Train = zeros(nS);
for i=1:nS     
    for j=1:nS 
        trans_i = 0;
        trans_i_j = 0; 
        % for all sequences
        for k=1:nQ
            for n=1:lQ-1 % ignore last state in seq (not a transition)
                if Train(k,n) == i 
                    trans_i = trans_i + 1;
                    if Train(k,n+1) == j 
                        trans_i_j = trans_i_j + 1;
                    end
                end
            end
        end
        A_Train(i,j) = trans_i_j / trans_i;
    end
end

% the observation sequence O = {S1, S1, S3, S3}. Its probability is
% P(O|A, pi) = P(S1) · P(S1|S1) · P(S3|S1) · P(S3|S3)
%           = pi1    · a11      · a13      · a33
Train_probs = [];
for k=1:nQ
    seq_prob = pi_Train(Train(k,1)); %single sequence probabiltiy
    for n=2:lQ
        seq_prob = seq_prob * A_Train( Train(k,n-1), Train(k,n) );
    end
    Train_probs = [Train_probs; seq_prob];
end
Train_likelihood = mean(Train_probs);

[nQ, lQ] = size(Valid);
Valid_probs = [];
for k=1:nQ
    seq_prob = pi_Train(Valid(k,1));
    for n=2:lQ
        seq_prob = seq_prob * A_Train( Valid(k,n-1), Valid(k,n) );
    end
    Valid_probs = [Valid_probs; seq_prob];
end
Valid_likelihood = mean(Valid_probs);

