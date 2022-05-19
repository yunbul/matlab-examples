clc
clear all

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

%divide this data in order to work easily
% class labels
 classes = data(:,2);
%training data
 trdata=data(1:40,:);   trclasses=classes(1:40);
% test data
 tsdata=data(41:80,:); tsclasses=classes(41:80);
%plot data
 figure('Name','INPUT & OUTPUT')
 set(gcf,'Position',[100 100 800 500])
 hold on
 plot(trdata(:,1),trdata(:,2),'+','Color', 'cyan');
 plot(tsdata(:,1),tsdata(:,2),'x','Color', 'blue');
 axis([0 7 -0.5 1.5])
%=====================================
[r,c]=size(data(:,1));
input = data;
desired_out = classes;

bias = -1;
coeff = 0.2; %learning rate
weights = rand(c+1,1); % or fill zeros(c+1,1);
%we will assume that weights(1,1) is for bias (+ w0) and weights(2,1) is for (w1 x)
iterations = 1000;
errors = zeros(iterations,1);
out = zeros(r,1);
plot_on = [1,5,10,20,40,80,150,250,500,1000];
plot_on_iter = 1;

for i=1:iterations
    out = zeros(r,1);
    for j=1:r
        y = input(j,1) * weights(2,1) + bias * weights(1,1); %resulting function
        out(j)= 1 / (1+exp(-y)); %apply sigmoid to get output between 0-1
        delta = desired_out(j) - out(j); %difference
        if delta > 0.0001
            errors(i,1) = errors(i,1) + 1;
        end
        weights(1,1) = weights(1,1) + coeff*bias*delta; %improve weights respectively
        weights(2,1) = weights(2,1) + coeff*input(j,1)*delta;
    end
    if plot_on_iter <= 10 && plot_on(1,plot_on_iter) == i %plot results on specified iterations
        %expected points (class labels) for all samples on plane (x,y)
        sigmoid_points = horzcat(input(:,1),out(:,1)); 
        %just sorted to draw the polyline that is connecting all points = sigmoid output
        sigmoid_points = sortrows(sigmoid_points,1);
        plot(sigmoid_points(:,1),sigmoid_points(:,2),'Color', 'red');
        %pick to points to draw seperator line : y = w1 x + w0
        liners = zeros(2); %we need 2 points on the plane to draw seperator line
        pys = [-0.5,1.5]; %where y = {1.5, -0.5}
        for p=1:2
            py = pys(1,p); px = (py - weights(1,1))/weights(2,1); %find corresponding x's
            %problem with bias(w0) -> shifted seperator lines to (+)x !
            liners(p,1) = px + 6.4; liners(p,2) = py; 
        end
        plot(liners(:,1),liners(:,2),'Color', 'green'); %plot seperator
        strmin = [num2str(i),'th'];
        text(liners(:,1),liners(:,2),strmin,'HorizontalAlignment','left');
        plot_on_iter = plot_on_iter + 1;
    end
end

hold off
figure('Name','ERROR RATE')
hold on
plot(errors,'-','Color', 'red');


