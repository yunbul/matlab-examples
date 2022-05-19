clc
clear all
close all

Train = load('d_reg_tra.txt');
Valid = load('d_reg_val.txt');
%plot without normalization
figure(998),hold on,
plot(Train(:,1),Train(:,2),'bo');
plot(Valid(:,1),Valid(:,2),'rx');
title('Fit Models'),legend(': tra',': val');

%normalize training data
[Train_norm, mu, sigma, dummy] = MeanStdNormalize(Train);

alpha = 0.025; %learning rate used in both kmeans and gradient-desc
num_iters = 300; %gradient-desc epoch limit
animateKmeans = 0; %switch(0/1) to animate kmeans for H={2,3,...,H_limit}
H_limit = 15; %max of H to display under/well/over-fit cases
[dummy, text_ptr_x_tra] = max(Train(:,1)); %used for leftarrow
[dummy, text_ptr_x_val] = max(Valid(:,1)); %used for leftarrow

RMSE_tra = []; RMSE_val = []; %[gradient iter | H | error] matrices to surf

%MAIN LOOP BEGIN
for H=5:5:H_limit %for different H's

%get cluster centroids
if animateKmeans == 1
    %call with H instead of 999 (dummy) to animate on discrete figures
    %call with an additional param to plot kmeans costs -> ..,alpha,H,x)
    [C] = OnlineKMeans(Train_norm,H,alpha,999,1001);
else
    [C] = OnlineKMeans(Train_norm,H,alpha); 
end

%get spread of each centroid (Cs : clusters)
[Sh, Cs] = GetSpread(Train_norm,C);

%get rbf outputs of hidden units and append bias
[G_tra] = GetGaussian(Train_norm(:,1),C(:,1),Sh);
[G_tra] = [G_tra ones(size(G_tra,1),1)];

theta = normrnd(0,1,H+1,1); %+1 for bias weight

%normalize valid set using same mean and std-dev from training
[dum1, dum2, dum3, Valid_norm] = MeanStdNormalize(0,Valid,mu,sigma);
%get hidden layer outputs
[G_val] = GetGaussian(Valid_norm(:,1),C(:,1),Sh);
[G_val] = [G_val ones(size(G_val,1),1)];

%update hidden weights (theta) with gradient-desc
%G_val and Valid_norm are just sent to get results at the same time
%they have nothing to do with update (see GradientDescent.m)
[theta, J_history_tra, J_history_val] = GradientDescent(G_tra, Train_norm(:,2), theta, alpha, num_iters, G_val ,Valid_norm(:,2));

%create [gradient iter | H | error] format to surf after main loop
[RMSE_tra] = vertcat(RMSE_tra, [(1:numel(J_history_tra))' ones(numel(J_history_tra),1)*H J_history_tra]);
[RMSE_val] = vertcat(RMSE_val, [(1:numel(J_history_val))' ones(numel(J_history_val),1)*H J_history_val]);

%get denormalized y predictions for training data 
[Predicts] = (G_tra*theta)*sigma(2) + mu(2);
Predicts = [Train(:,1) Predicts];
text_ptr_y_tra = Predicts(text_ptr_x_tra,2);
Predicts = sortrows(Predicts,1); %just to plot fit model (zigzag)
if H == H_limit %for final H
    figure(998),plot(Predicts(:,1),Predicts(:,2),'-b'); 
    text(Train(text_ptr_x_tra,1),text_ptr_y_tra,strcat('\leftarrow ','H:',num2str(H)),'color','b');
else
    if mod(H,5) == 0 %for intermediate H values
    figure(998),plot(Predicts(:,1),Predicts(:,2),'-c'); 
    text(Train(text_ptr_x_tra,1),text_ptr_y_tra,strcat('\leftarrow ','H:',num2str(H)),'color','c');
    end
end

%get denormalized y predictions for validation data 
[Predicts] = (G_val*theta)*sigma(2) + mu(2);
Predicts = [Valid(:,1) Predicts];
text_ptr_y_val = Predicts(text_ptr_x_val,2);
Predicts = sortrows(Predicts,1);
if H == H_limit
    figure(998),plot(Predicts(:,1),Predicts(:,2),'-r');
    text(Valid(text_ptr_x_val,1),text_ptr_y_val,strcat('\leftarrow ','H:',num2str(H)),'color','r');
else
    if mod(H,5) == 0
    figure(998),plot(Predicts(:,1),Predicts(:,2),'-y');
    text(Valid(text_ptr_x_val,1),text_ptr_y_val,strcat('\leftarrow ','H:',num2str(H)),'color','y');
    end
end

end
%MAIN LOOP END

%plot surfed J_history for every grad-desc
%to display 'nH' <-> 'error' relation
x = RMSE_tra(:,1)./10;
y = RMSE_tra(:,2);
z = RMSE_tra(:,3);
x_edge = [floor(min(x)):1:ceil(max(x))];
y_edge = [floor(min(y)):1:ceil(max(y))];
[X,Y] = meshgrid(x_edge,y_edge);
Z = griddata(x,y,z,X,Y);
figure(2001)
title('Train Surf RMSE for H Incerements over Epochs')
hold on,xlabel('gradient epochs (x10)'),ylabel('num of hidden nodes'),zlabel('RMSE')
surf(X,Y,Z);

x = RMSE_val(:,1)./10;
y = RMSE_val(:,2);
z = RMSE_val(:,3);
x_edge = [floor(min(x)):1:ceil(max(x))];
y_edge = [floor(min(y)):1:ceil(max(y))];
[X,Y] = meshgrid(x_edge,y_edge);
Z = griddata(x,y,z,X,Y);
figure(2002)
title('Valid Surf RMSE for H Incerements over Epochs')
hold on,xlabel('gradient epochs (x10)'),ylabel('num of hidden nodes'),zlabel('RMSE')
surf(X,Y,Z);

