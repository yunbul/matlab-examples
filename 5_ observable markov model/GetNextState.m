function [S] = GetNextState(p)

%  generates a random state label (index) S:{1, 2, ..., length(p)} 
%  with given state probability distribution 'p'

u = rand;
i = 1;
s = p(1);

while (u > s) && (i < length(p))
  i=i+1;
  s=s+p(i);
end

S=i;
end

