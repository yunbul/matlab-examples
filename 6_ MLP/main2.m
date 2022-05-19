clc
clear
%read data
Train = textread('d_reg_tra.txt'); %train
Valid = textread('d_reg_val.txt'); %validation
%get data size
[nT, dummy1] = size(Train);
[nV, dummy2] = size(Valid);

%sigmoid function definition
sigmoid = @(a)(1 / (1 + exp(-a)));

%learning rate
n = 0.025;
%num of epochs (loops over train data)
epochs = 300;
%plot fit model on every mod(epoch_k,plot_mod_div)=0
plot_mod_div = 50; 

%following switches defined to plot required results (1/0 for on/off)
plot_on_diff_H = 1; %plot fitted model curve on tra/val for every H incerement
plot_last_epochs_fit = 1; %plot fit model just at last epochs (ignored if plot_on_diff_H = 0)
plot_wtd_sigmoid_outs = 1; %plot weighted sigmoid outputs of hiddens separately at last epoch (ignored if plot_on_diff_H = 0)

plot_discrete_err_on_H = 1; %plot curve graph of tra/val errors (separate)
plot_abs_err_surfs = 1; %plot surf (3D version of curves) of tra/val errors (continuous)

%initial random weights normal distribution params
mean = 0;
std_dev = 1;
%following used to plot/surf error results (..._all used to surf)
%MAE : mean-abs-err, RMSE : mean-sqrt-err
MAE_tra = zeros(epochs,3); MAE_tra_all = zeros(0,3);
RMSE_tra = zeros(epochs,3); RMSE_tra_all = zeros(0,3);
MAE_val = zeros(epochs,3); MAE_val_all = zeros(0,3);
RMSE_val = zeros(epochs,3); RMSE_val_all = zeros(0,3);

if plot_discrete_err_on_H == 1
    figure(1000),hold on,xlabel('epochs'),ylabel('num of hidden nodes'),zlabel('MAE')
    title('Discrete Mean Absolute Error for H Incerements')
    figure(2000),hold on,xlabel('epochs'),ylabel('num of hidden nodes'),zlabel('RMSE')
    title('Discrete Root Mean Square Error for H Incerements')
end

%for different num of hidden neurons
for H=5:5:15
	
	W = normrnd(mean,std_dev,H,1); %input weights to hidden neurons
	W_bias = normrnd(mean,std_dev,H,1); %input bias weights to hidden neurons
	
	V = normrnd(mean,std_dev,H,1); %hidden weights to output
	V_bias = normrnd(mean,std_dev); %hidden bias weight to output
	
    if plot_on_diff_H == 1
        figure(H)
        hold on
        title(strcat('Tra/Val Fit Models with Intermediate Epochs when H=',num2str(H)))
        plot(Train(:,1),Train(:,2),'bx ',Valid(:,1),Valid(:,2),'r. ');
        legend(': tra',': val')
    end
	plotter_tra = zeros(nT,2);
	plotter_val = zeros(nV,2);
    if plot_wtd_sigmoid_outs == 1
        plotter_sigmoid_tra = Train(:,1);
        plotter_sigmoid_tra_Zs = zeros(nT,0);
        plotter_sigmoid_val = Valid(:,1);
        plotter_sigmoid_val_Zs = zeros(nV,0);
    end
    
	for k=1:epochs
	
		%back-propagating update calc
		for i=1:nT
			X = Train(i,1); %input
			Y = Train(i,2); %actual output
		    Z = arrayfun(sigmoid,(W.*X) + W_bias); %sigmoid outputs
		    prediction = sum(Z.*V) + V_bias;
		    diff = Y - prediction;
		    
		    delta_v = Z.*diff.*n;
		    delta_v_b = diff * n;
		    
		    delta_w =  (V.*diff).*Z.*(1-Z).*X.*n;
		    delta_w_b = (V.*diff).*Z.*(1-Z).*n;
		    
		    V = V + delta_v;
		    V_bias = V_bias + delta_v_b;
		    W = W + delta_w;
		    W_bias = W_bias + delta_w_b;
		end
	
		%overall train set error calcs
		MAE_tra_sum = 0;
		RMSE_tra_sum = 0;
		for j=1:nT
		    Z_out = arrayfun(sigmoid,(Train(j,1).*W) + W_bias);
            WZ_out = Z_out.*V;
		    prediction_out = sum(WZ_out) + V_bias;
		    if mod(k,plot_mod_div) == 0
	            plotter_tra(j,1) = Train(j,1);
				plotter_tra(j,2) = prediction_out;
                if k == epochs && plot_wtd_sigmoid_outs==1
                    plotter_sigmoid_tra_Zs = vertcat(plotter_sigmoid_tra_Zs,horzcat(WZ_out',V_bias));
                end
	        end
		    MAE_tra_sum = MAE_tra_sum + abs(Train(j,2)-prediction_out);
			RMSE_tra_sum = RMSE_tra_sum + (Train(j,2)-prediction_out)^2;
		end
		MAE_tra(k,1) = k; 
		MAE_tra(k,2) = H;
        MAE_tra(k,3) = MAE_tra_sum / nT;
		RMSE_tra(k,1) = k;
		RMSE_tra(k,2) = H;
        RMSE_tra(k,3) = sqrt(RMSE_tra_sum/nT);
        
		if mod(k,plot_mod_div) == 0 && plot_on_diff_H == 1
	        plotter_tra = sortrows(plotter_tra,1);
	        if k == epochs
                figure(H)
	            if plot_wtd_sigmoid_outs == 1
                    plotter_sigmoid_tra = horzcat(plotter_sigmoid_tra,plotter_sigmoid_tra_Zs);
                    plotter_sigmoid_tra = sortrows(plotter_sigmoid_tra,1);
                    for zi=1:1:H
                        plot(plotter_sigmoid_tra(:,1),plotter_sigmoid_tra(:,zi+1),'k+-');
                        text(plotter_sigmoid_tra(nT,1),plotter_sigmoid_tra(nT,zi+1),strcat('\leftarrow ','H',num2str(zi)),'color','k')
                    end
                    zi = zi + 1;
                    plot(plotter_sigmoid_tra(:,1),plotter_sigmoid_tra(:,zi+1),'k+-');
                    text(plotter_sigmoid_tra(nT,1),plotter_sigmoid_tra(nT,zi+1),'\leftarrow HB','color','k')
                end
                plot(plotter_tra(:,1),plotter_tra(:,2),'b+-');
                text(plotter_tra(nT,1),plotter_tra(nT,2),'\leftarrow Tfin','color','b')
            elseif plot_last_epochs_fit ~= 1
                figure(H)
	            leg_h_tra = plot(plotter_tra(:,1),plotter_tra(:,2),'c+-');
                text(plotter_tra(nT,1),plotter_tra(nT,2),strcat('\leftarrow Te',num2str(k)),'color','c')
            end
	    end
	    
	    %overall valid set error calcs
		MAE_val_sum = 0;
		RMSE_val_sum = 0;
		for j=1:nV
		    Z_out = arrayfun(sigmoid,(Valid(j,1).*W) + W_bias);
            WZ_out = Z_out.*V;
		    prediction_out = sum(WZ_out) + V_bias;
		    if mod(k,plot_mod_div) == 0
	            plotter_val(j,1) = Valid(j,1);
				plotter_val(j,2) = prediction_out;
                if k == epochs && plot_wtd_sigmoid_outs==1
                    plotter_sigmoid_val_Zs = vertcat(plotter_sigmoid_val_Zs,horzcat(WZ_out',V_bias));
                end
	        end
		    MAE_val_sum = MAE_val_sum + abs(Valid(j,2)-prediction_out);
			RMSE_val_sum = RMSE_val_sum + (Valid(j,2)-prediction_out)^2;
		end
		MAE_val(k,1) = k; 
        MAE_val(k,2) = H; 
		MAE_val(k,3) = MAE_val_sum / nV;
		RMSE_val(k,1) = k;
        RMSE_val(k,2) = H;
		RMSE_val(k,3) = sqrt(RMSE_val_sum/nV);
		if mod(k,plot_mod_div) == 0 && plot_on_diff_H == 1
	        plotter_val = sortrows(plotter_val,1);
	        if k == epochs
                figure(H)
	            if plot_wtd_sigmoid_outs == 1
                    plotter_sigmoid_val = horzcat(plotter_sigmoid_val,plotter_sigmoid_val_Zs);
                    plotter_sigmoid_val = sortrows(plotter_sigmoid_val,1);
                    for zi=1:1:H
                        plot(plotter_sigmoid_val(:,1),plotter_sigmoid_val(:,zi+1),'go-');
                        text(plotter_sigmoid_val(nV,1),plotter_sigmoid_val(nV,zi+1),strcat('\leftarrow ','H',num2str(zi)),'color','g')
                    end
                    zi = zi + 1;
                    plot(plotter_sigmoid_val(:,1),plotter_sigmoid_val(:,zi+1),'go-');
                    text(plotter_sigmoid_val(nV,1),plotter_sigmoid_val(nV,zi+1),'\leftarrow HB','color','g')
                end
                plot(plotter_val(:,1),plotter_val(:,2),'ro-');
                text(plotter_val(nV,1),plotter_val(nV,2),'\leftarrow Vfin','color','r')
            elseif plot_last_epochs_fit ~= 1
                figure(H)
	            leg_h_val = plot(plotter_val(:,1),plotter_val(:,2),'yo-');
                text(plotter_val(nV,1),plotter_val(nV,2),strcat('\leftarrow Ve',num2str(k)),'color','y')
	        end
		end
	
    end
    
    if plot_discrete_err_on_H == 1
    figure(1000)
	plot3(MAE_tra(:,1),MAE_tra(:,2),MAE_tra(:,3),'b-');
    plot3(MAE_val(:,1),MAE_val(:,2),MAE_val(:,3),'r-');
    legend(': tra',': val')
    
    figure(2000)
	leg_err_tra = plot3(RMSE_tra(:,1),RMSE_tra(:,2),RMSE_tra(:,3),'b-');
	leg_err_val = plot3(RMSE_val(:,1),RMSE_val(:,2),RMSE_val(:,3),'r-');
    legend(': tra',': val')
    end
    if plot_abs_err_surfs == 1
    MAE_tra_all = vertcat(MAE_tra_all,MAE_tra);
    MAE_val_all = vertcat(MAE_val_all,MAE_val);
    RMSE_tra_all = vertcat(RMSE_tra_all,RMSE_tra);
    RMSE_val_all = vertcat(RMSE_val_all,RMSE_val);
    end
end

if plot_abs_err_surfs == 1
x = MAE_tra_all(:,1);
y = MAE_tra_all(:,2);
z = MAE_tra_all(:,3);
x_edge = [floor(min(x)):1:ceil(max(x))];
y_edge = [floor(min(y)):1:ceil(max(y))];
[X,Y] = meshgrid(x_edge,y_edge);
Z = griddata(x,y,z,X,Y);
figure(1001)
title('Train Surf MAE for H Incerements over Epochs')
hold on,xlabel('epochs'),ylabel('num of hidden nodes'),zlabel('MAE')
surf(X,Y,Z);

x = MAE_val_all(:,1);
y = MAE_val_all(:,2);
z = MAE_val_all(:,3);
x_edge = [floor(min(x)):1:ceil(max(x))];
y_edge = [floor(min(y)):1:ceil(max(y))];
[X,Y] = meshgrid(x_edge,y_edge);
Z = griddata(x,y,z,X,Y);
figure(1002)
title('Valid Surf MAE for H Incerements over Epochs')
hold on,xlabel('epochs'),ylabel('num of hidden nodes'),zlabel('MAE')
surf(X,Y,Z);

x = RMSE_tra_all(:,1);
y = RMSE_tra_all(:,2);
z = RMSE_tra_all(:,3);
x_edge = [floor(min(x)):1:ceil(max(x))];
y_edge = [floor(min(y)):1:ceil(max(y))];
[X,Y] = meshgrid(x_edge,y_edge);
Z = griddata(x,y,z,X,Y);
figure(2001)
title('Train Surf RMSE for H Incerements over Epochs')
hold on,xlabel('epochs'),ylabel('num of hidden nodes'),zlabel('RMSE')
surf(X,Y,Z);

x = RMSE_val_all(:,1);
y = RMSE_val_all(:,2);
z = RMSE_val_all(:,3);
x_edge = [floor(min(x)):1:ceil(max(x))];
y_edge = [floor(min(y)):1:ceil(max(y))];
[X,Y] = meshgrid(x_edge,y_edge);
Z = griddata(x,y,z,X,Y);
figure(2002)
title('Valid Surf RMSE for H Incerements over Epochs')
hold on,xlabel('epochs'),ylabel('num of hidden nodes'),zlabel('RMSE')
surf(X,Y,Z);
end

