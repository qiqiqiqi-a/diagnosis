clc
clear
load origin_data.txt
load data1.txt;
data = data1;
data_out = origin_data(:,10:end);
new_data = zeros(size(data_out));
for i = 1:size(data_out, 1)
    [~, idx] = max(data_out(i, :)); 
    new_data(i, idx) = 1;
end
Newdata_out = new_data ;
count_ones = sum(new_data, 1);
column_10 = Newdata_out(:, 1);%%选择当前子置信规则库
data(:, 10) = column_10;
sum1 = sum(data(:,10) ==1 );
sum0 = length(data(:,10)) - sum1;
lb_bound = [0 0 0 0 0 0 0 0 0];
ub_bound = [1 1 1 1 1 1 1 1 1];
num_rules = 8;
num_att = 9; 
num_scales = 2;
uscale = [0 1];
length = num_rules * (num_att + 1 + num_scales); 
num_indi = 50; 
gen_all = 200;
gt_all_runs = zeros(30,172, 2);
for run =1:30

     shuffledData = data(randperm(size(data,1)),:);
     train_set = shuffledData(1:172,:);  
     test_set = shuffledData(1:172,:); 

    [error_GA(run) brb_GA(run,1:length) best_GA(run, 1:gen_all)] = return_GA(train_set, num_indi, gen_all, num_rules, num_att, num_scales, ub_bound, lb_bound, uscale);
    [error_test_GA(run), predict_result(run,:),gt_all_runs(run,:, :),T1(run,:), F1(run,:), T0(run,:), F0(run,:)] = mse_chang_pipeline_test(brb_GA(run,1:length), test_set(:,1:num_att), test_set(:, num_att + 1),  num_rules, num_scales, uscale); 
  
    num_1(run) = sum(test_set(:,end)' == 1);
   num_0(run)= sum(test_set(:,end)' == 0);
   total_samples(run) = num_0(run)+ num_1(run);
   acc(run)= 1- error_test_GA(run);
   TP_acc(run) = T1(run,:)/num_1(run);
   FP_acc(run) = F1(run,:)/num_1(run);
   TN_acc(run) = T0(run,:)/num_0(run);
   FN_acc(run) = F0(run,:)/num_0(run);
end
save result.mat
 

