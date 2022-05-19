function [ G ] = GetGaussian( X,C,Sh )

%rbf : p = exp( -(  |x-m|^2 / (2Sh)^2  )  )
%                     |__ euclid^2 = sqrt( sum(( X(i,:)-C(j,:) ).^2,2) )^2
%                                  =       sum(( X(i,:)-C(j,:) ).^2,2)


G = [];
for i=1:size(X,1)
	for j=1:size(C,1)
    	G(i,j) = exp ( - sum((X(i,:)-C(j,:)).^2,2) / (2*(Sh(j)^2)) );
	end
end

end

