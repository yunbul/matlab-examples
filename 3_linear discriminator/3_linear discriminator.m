clc
clear all
%cd c:\Users\yunus.onur\....
%load data.txt

data = [
  2.5493025e+000  1.0000000e+000
  1.8474957e+000  1.0000000e+000
  2.1501557e+000  1.0000000e+000
  2.3746311e+000  1.0000000e+000
  1.5424579e+000  1.0000000e+000
  1.9739042e+000  1.0000000e+000
  1.8867639e+000  1.0000000e+000
  2.3246809e+000  1.0000000e+000
  2.4919711e+000  1.0000000e+000
  2.2748111e+000  1.0000000e+000
  1.7797506e+000  1.0000000e+000
  1.7437455e+000  1.0000000e+000
  1.2718867e+000  1.0000000e+000
  2.0037521e+000  1.0000000e+000
  2.8940945e+000  1.0000000e+000
  2.0935274e+000  1.0000000e+000
  2.0129895e+000  1.0000000e+000
  1.5877104e+000  1.0000000e+000
  2.2097806e+000  1.0000000e+000
  1.9609595e+000  1.0000000e+000
  4.2082362e+000  0.0000000e+000
  5.6834167e+000  0.0000000e+000
  4.2556886e+000  0.0000000e+000
  5.1296513e+000  0.0000000e+000
  5.3864894e+000  0.0000000e+000
  4.5975455e+000  0.0000000e+000
  5.5177101e+000  0.0000000e+000
  5.4211538e+000  0.0000000e+000
  4.2620661e+000  0.0000000e+000
  4.7340086e+000  0.0000000e+000
  5.0469403e+000  0.0000000e+000
  5.3733509e+000  0.0000000e+000
  4.1606475e+000  0.0000000e+000
  4.4845986e+000  0.0000000e+000
  5.1101185e+000  0.0000000e+000
  5.2602125e+000  0.0000000e+000
  5.6969615e+000  0.0000000e+000
  5.4712884e+000  0.0000000e+000
  5.0373411e+000  0.0000000e+000
  4.3963587e+000  0.0000000e+000
  2.2264294e+000  1.0000000e+000
  1.3957472e+000  1.0000000e+000
  1.4503037e+000  1.0000000e+000
  1.3071948e+000  1.0000000e+000
  1.4611502e+000  1.0000000e+000
  2.4442452e+000  1.0000000e+000
  1.6048851e+000  1.0000000e+000
  2.0275595e+000  1.0000000e+000
  2.5379102e+000  1.0000000e+000
  1.1093012e+000  1.0000000e+000
  1.4328975e+000  1.0000000e+000
  2.2021406e+000  1.0000000e+000
  2.5002854e+000  1.0000000e+000
  2.1914232e+000  1.0000000e+000
  2.1216386e+000  1.0000000e+000
  2.0751345e+000  1.0000000e+000
  1.7728950e+000  1.0000000e+000
  2.2417593e+000  1.0000000e+000
  2.0864224e+000  1.0000000e+000
  1.4500683e+000  1.0000000e+000
  5.0500758e+000  0.0000000e+000
  5.1714619e+000  0.0000000e+000
  5.5217232e+000  0.0000000e+000
  5.3055895e+000  0.0000000e+000
  4.5832676e+000  0.0000000e+000
  5.0145115e+000  0.0000000e+000
  3.3946688e+000  0.0000000e+000
  5.6900929e+000  0.0000000e+000
  4.6938376e+000  0.0000000e+000
  4.6586182e+000  0.0000000e+000
  5.0748844e+000  0.0000000e+000
  4.6434275e+000  0.0000000e+000
  4.4487817e+000  0.0000000e+000
  4.4742676e+000  0.0000000e+000
  4.7436141e+000  0.0000000e+000
  4.6696199e+000  0.0000000e+000
  5.0610120e+000  0.0000000e+000
  4.5697072e+000  0.0000000e+000
  6.1361912e+000  0.0000000e+000
  4.4242928e+000  0.0000000e+000
    ];

w = rand * 0.05; %the slope starting with random value
w0 = rand * 0.05; %the intersection point

[n N] = size(data);

%sigmoid function
sigmoid = @(a)(1 / (1 + exp(-a)));

lf = 0.2; %learning factor

nTR = 40; %number of training instances
nTS = 40; %number of test instances
epoch = 1000;

figure,
hold on,
for i=1:nTR
	if data(i, 2)==1
		plot(data(i,1),data(i,2),'rx ');axis([1 6.3 -1 2]);
	else
		plot(data(i,1),data(i,2),'bo ');axis([1 6.3 -1 2]);
	end
end

%m is epoch
y = zeros(nTR, 1); %the output of sigmoid function on training set

errorTr = zeros(1,epoch); %training error on each epoch
errorTs = zeros(1,epoch); %testing error on each epoch

slopes = zeros(1, epoch);
intersects = zeros(1, epoch);

for m=1:epoch
	
	%for each example
	for t=1:nTR 
		y(t) = sigmoid(w * data(t,1) + w0);
		delta_w = lf * (data(t,2) - y(t)) * data(t,1);
		delta_w0 = lf * (data(t,2) - y(t));
		w = w + delta_w;
		w0 = w0 + delta_w0;
	end
	
	if m==5 | m==10 | m==25 | m==50 | m==100 | m==500 | m==1000
		m
		x1 = 1;
		x2 = 7;
		y1 = w * x1 + w0;
		y2 = w * x2 + w0;
		plot([x1 x2],[y1 y2],'g'); %line
		pause

		xx = 1.1:0.130:6.2;
		plot(xx',y,'r'); %output of sigmoid
		pause
	end
	
	predictedTr = zeros(nTR,1); %error on training set
	
	for k=1:nTR
		if y(k) > 0.5
			predictedTr(k,1) = 1;
		else
			predictedTr(k,1) = 0;
		end
	end		
	
	for i=1:nTR
		if data(i,2) ~= predictedTr(i,1)
			errorTr(m) = errorTr(m) + 1;
		end
	end

	%error on test set
	predictedTs = zeros(nTS,1);
	for k=1:nTS
		predictedTs(k) = sigmoid(w * data(k + nTR,1) + w0);
		if predictedTs(k) > 0.5
			predictedTs(k) = 1;
		else
			predictedTs(k) = 0;
		end
	end
	
	for i=1:nTS
		if data(i+nTR,2) ~= predictedTs(i,1)
			errorTs(m) = errorTs(m) + 1;
		end
	end

	slopes(m) = w;
	intersects(m) = w0;

	lf = lf * 0.995;
end

plot3(slopes,intersects,errorTr,'b',slopes,intersects,errorTs,'r');
