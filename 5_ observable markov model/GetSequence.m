function [Q] = GetSequence(pi, A, len)

%  generates a markov chain sequence of length 'len'
%  which is a time series of states {1, 2, 3} 
%  with given initial probabilities 'pi' and the transition matrix 'A'

Q=zeros(1,len+1); % sequence
Q(1)=GetNextState(pi); % generate first (at time 0)

for i=1:len,
  Q(i+1) = GetNextState(A(Q(i),:));
end

% % plot sequence
% t=0:len; %time series
% plot(t, Q, '*');
% axis([0 len 0 (length(pi)+1)]);
end