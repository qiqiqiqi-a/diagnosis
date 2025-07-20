
function [globalBestValue, globalBestIndividual, trace] = return_GA(btr_data, np, ng, number_rules, num_att, num_scales, ub_bound, lb_bound, uscale)

Nind = np; 
input = btr_data(:, 1:num_att);  
output = btr_data(:, num_att+1);
size_input = size(input);  

numVariables = number_rules *(size_input(1,2) +1 + num_scales);

for i=1:size_input(1,2)
    
    lb(number_rules*(i-1) +1:number_rules * i) = lb_bound(i); 
    ub(number_rules*(i-1) +1:number_rules * i) = ub_bound(i); 
    
end

lb(number_rules * size_input(1,2)+1 : numVariables) = 0; 
ub(number_rules * size_input(1,2)+1 : numVariables) = 1;


pop = initializePop_pipeline(np, lb_bound, ub_bound, lb,ub,number_rules,numVariables,num_scales); %20×50

for i=1:Nind
    [objv(i),predict_result(i,:)] = mse_chang_pipeline_s(pop(i,:), input, output,  number_rules, num_scales, uscale);   
end
    
for gen =1:ng  
    
    if mod(gen,100) ==0
    
    gen
    
    end 
    
       crossoverProb = 0.9;
  
   newPop = xovsp(pop, crossoverProb); 

    newPop = mut(newPop); 


     for k=1:np  
        for j=1:numVariables         
            if (newPop(k,j)<lb(1,j))&&(newPop(k,j)>=(2*lb(1,j)-ub(1,j)))
                newPop(k,j)=lb(1,j)+(lb(1,j)-newPop(k,j));
            end
            if (newPop(k,j)<=(2*ub(1,j)-lb(1,j)))&&(newPop(k,j)>ub(1,j))
                newPop(k,j)=ub(1,j)-(newPop(k,j)-ub(1,j));
            end
            if newPop(k,j)<(2*lb(1,j)-ub(1,j))
                newPop(k,j)=lb(1,j)+rand*(ub(1,j)-lb(1,j));
            end
            if newPop(k,j)>(2*ub(1,j)-lb(1,j))
                newPop(k,j)=lb(1,j)+rand*(ub(1,j)-lb(1,j));
            end    
        end
       
     end

    for ii=1:size_input(1,2) %4
    
        newPop(:,number_rules*(ii-1) +1) = lb_bound(ii);
        newPop(:,number_rules*ii ) = ub_bound(ii);

   
    end
        
    newPop(:,number_rules * (size_input(2) +1) +1:numVariables) = y_normalize(newPop(:,number_rules *(size_input(2) +1)+1:numVariables),num_scales);
 
    for i=1:Nind
        [newObjv(i),newpredict_result(i,:)] = mse_chang_pipeline_s(newPop(i,:), input, output,  number_rules, num_scales, uscale);   
    end
   
   allObjv = [objv, newObjv]; 
   allPop = [pop; newPop];
   [fitness, index] = sort(allObjv);
   pop = allPop(index(1:Nind),:);
   objv = allObjv(index(1:Nind));

   trace(gen) = min(objv);
   globalBestIndividual = pop(1,:);

   globalBestValue = trace(gen)
end
end


%% initializePop_pipeline
function y=initializePop_pipeline(np,lb_bound, ub_bound, lb,ub,number_rules,numVariables,num_scales)  %lq为起点中变量位的个数
y=zeros(np,numVariables);   
for i=1:length(lb)
    
    y(:,i)=lb(i)+(ub(i)-lb(i))*rand(np,1);  
end
for i=1:length(lb_bound)  %1-4
    
    y(:,number_rules*(i-1) +1) = lb_bound(i);
    y(:,number_rules*i) = ub_bound(i);
end
y(:,(number_rules * (length(lb_bound) +1)+1):numVariables) = y_normalize(y(:, (number_rules * (length(lb_bound) +1) +1):numVariables),num_scales);
end

%%  mse_chang_pipeline_s
function [error, predict_result] = mse_chang_pipeline_s(individual, input, output, number_rules, num_scales, uscale)
    size_input = size(input);
    error = 0;
    for i = 1:size_input(1, 1)
        [rules_activated, weights_activated] = determine_rule_pipeline(input(i,:), individual(1:(size_input(1, 2)) * number_rules), individual(((size_input(1, 2)) * number_rules + 1):(size_input(1, 2) + 1) * number_rules));
        scales_activated = zeros(length(rules_activated), num_scales);
        for j = 1:length(rules_activated)
            scales_activated(j, :) = individual((size_input(1, 2) + 1) * number_rules + num_scales * rules_activated(j) - (num_scales - 1):(size_input(1, 2) + 1) * number_rules + num_scales * rules_activated(j));
        end
        gt(i, :) = return_gt(weights_activated, scales_activated);
        [max_val, max_idx] = max(gt(i, :));  % 找出最大值及其索引
        single_error(i) = uscale(max_idx);
    end
    predict_result = single_error;
    num_errors = sum(predict_result ~= output');
    total_samples = length(output);
    error = num_errors / total_samples; 
    num_samples_minority = sum(output == 1);  
    num_samples_majority = sum(output == 0);  
    error_minority= FN / (num_samples_minority + eps);
    w_m = num_samples_minority / total_samples; 
    w_m = log(1+(1-w_m));
     error = error +  w_m *error_minority ;
end

%% xovsp
function NewChrom = xovsp(OldChrom, XOVR);

if nargin < 2, XOVR = NaN; end


   NewChrom = xovmp(OldChrom, XOVR, 1, 0);
end
function NewChrom = xovmp(OldChrom, Px, Npt, Rs);

[Nind,Lind] = size(OldChrom);

if Lind < 2, NewChrom = OldChrom; return; end

if nargin < 4, Rs = 0; end
if nargin < 3, Npt = 0; Rs = 0; end
if nargin < 2, Px = 0.7; Npt = 0; Rs = 0; end
if isnan(Px), Px = 0.7; end
if isnan(Npt), Npt = 0; end
if isnan(Rs), Rs = 0; end
if isempty(Px), Px = 0.7; end
if isempty(Npt), Npt = 0; end
if isempty(Rs), Rs = 0; end

Xops = floor(Nind/2);
DoCross = rand(Xops,1) < Px;
odd = 1:2:Nind-1;
even = 2:2:Nind;

Mask = ~Rs | (OldChrom(odd, :) ~= OldChrom(even, :));
Mask = cumsum(Mask')';

xsites(:, 1) = Mask(:, Lind);
if Npt >= 2,
        xsites(:, 1) = ceil(xsites(:, 1) .* rand(Xops, 1));
end
xsites(:,2) = rem(xsites + ceil((Mask(:, Lind)-1) .* rand(Xops, 1)) ...
                                .* DoCross - 1 , Mask(:, Lind) )+1;

% Express cross sites in terms of a 0-1 mask
Mask = (xsites(:,ones(1,Lind)) < Mask) == ...
                        (xsites(:,2*ones(1,Lind)) < Mask);

if ~Npt,
        shuff = rand(Lind,Xops);
        [ans,shuff] = sort(shuff);
        for i=1:Xops
          OldChrom(odd(i),:)=OldChrom(odd(i),shuff(:,i));
          OldChrom(even(i),:)=OldChrom(even(i),shuff(:,i));
        end
end

NewChrom(odd,:) = (OldChrom(odd,:).* Mask) + (OldChrom(even,:).*(~Mask));
NewChrom(even,:) = (OldChrom(odd,:).*(~Mask)) + (OldChrom(even,:).*Mask);

if rem(Nind,2),
  NewChrom(Nind,:)=OldChrom(Nind,:);
end

if ~Npt,
        [ans,unshuff] = sort(shuff);
        for i=1:Xops
          NewChrom(odd(i),:)=NewChrom(odd(i),unshuff(:,i));
          NewChrom(even(i),:)=NewChrom(even(i),unshuff(:,i));
        end
end

end
%% mut
function NewChrom = mut(OldChrom,Pm,BaseV)

[Nind, Lind] = size(OldChrom) ;

if nargin < 2, Pm = 0.7/Lind ; end
if isnan(Pm), Pm = 0.7/Lind; end

if (nargin < 3), BaseV = crtbase(Lind);  end
if (isnan(BaseV)), BaseV = crtbase(Lind);  end
if (isempty(BaseV)), BaseV = crtbase(Lind);  end

if (nargin == 3) & (Lind ~= length(BaseV))
   error('OldChrom and BaseV are incompatible'), end

BaseM = BaseV(ones(Nind,1),:) ;

NewChrom = rem(OldChrom+(rand(Nind,Lind)<Pm).*ceil(rand(Nind,Lind).*(BaseM-1)),BaseM);
end

%% y_normalize
function y_new = y_normalize(y,num_scales)

    size_y = size(y);
    
for i = 1: size_y(1,2)/num_scales
    
    total(:,1) = zeros(size_y(1,1),1);
    
   for j =1: num_scales
    
       total = total + y(:,(i-1) *num_scales + j);
    
   end    
    
   
   for j =1: num_scales
    
       y_new(:, (i-1) *num_scales + j) = y(:,(i-1) *num_scales + j)./total;
    
   end
       
end
end
%% determine_rule_pipeline
function [rules ,weights] = determine_rule_pipeline(input, individual,weights_original)
size_input = size(input);
abstract = zeros(length(input), length(individual)/length(input));
for i = 1:length(input)   
   abstract(i,:) = input(i) - individual((length(individual)/length(input)* (i-1) + 1): (length(individual)/length(input)* i)); 
   if i ==1  
       if max(abstract(i,:)) ==0 || min(abstract(i,:)) ==0
           rule_activated(1) = find(abstract(i,:) == 0);
           weights_activated(1) = 1;
   
       else 
             rule_activated(1) = find(abstract(i,:) == min(abstract(i,(find(abstract(i,:)>=0))))); 
           rule_activated(2) = find(abstract(i,:) == max(abstract(i,(find(abstract(i,:)< 0)))));
                gap = abs(individual (rule_activated(length(rule_activated) -1) + length(individual)/length(input) *(i-1)) - individual (rule_activated(length(rule_activated) ) + length(individual)/length(input) *(i-1)));
                 weights_activated(length(rule_activated)- 1) = abs(input(i) - individual( rule_activated( length(rule_activated) )+length(individual)/length(input) *(i-1)) )/gap;
           weights_activated(length(rule_activated)) = abs(input(i) - individual( rule_activated( length(rule_activated) -1 )+length(individual)/length(input) *(i-1)) )/gap;
       end
   else
          if max(abstract(i,:)) ==0 || min(abstract(i,:)) ==0
                  rule_activated(length(rule_activated) +1) = find(abstract(i,:) == 0);
           weights_activated(length(rule_activated) ) = 1;
          else
                  rule_activated(length(rule_activated) + 1) = find(abstract(i,:) == min(abstract(i, (find(abstract(i,:)>=0)))));
           rule_activated(length(rule_activated) + 1) = find(abstract(i,:) == max(abstract(i, (find(abstract(i,:)< 0)))));
              gap = abs(individual (rule_activated(length(rule_activated) -1) + length(individual)/length(input) *(i-1)) - individual (rule_activated(length(rule_activated) ) + length(individual)/length(input) *(i-1)));
              weights_activated(length(rule_activated)- 1) = abs(input(i) - individual( rule_activated( length(rule_activated) )+length(individual)/length(input) *(i-1)) )/gap;
           weights_activated(length(rule_activated)) = abs(input(i) - individual( rule_activated( length(rule_activated) -1 )+length(individual)/length(input) *(i-1)) )/gap;
             end
   end 
end
weights = zeros(length(individual)/length(input),1); 
for i=1:length(rule_activated)
    
    for j=1:length(individual)/length(input) 
        
        if rule_activated(i) == j
            
           weights(j) = weights(j) + weights_activated(i);
            
           break
        end            
    end
end

for i =1: length(weights)
       
    rules(i) = i;
    weights(i) = weights(i) * weights_original(i);
    
end
total_weight = sum(weights);

weights = weights./total_weight;

weight_count = length(individual)/length(input);

count =0;

for i = 1: weight_count
    
    if weights(i - count) ==0

        rules(i - count) = [];
        weights(i - count) = [];  
        
        count = count +1;
        
    end

end
end

%% 

function gt = return_gt(weights_activated, scales_activated)
if length(weights_activated) == 1
    gt = scales_activated; 
else
    size_scales = size(scales_activated);
       m = zeros(length(weights_activated), size_scales(1,2));
    m_R = zeros(length(weights_activated));
    m_R_ = zeros(length(weights_activated));
    for i =1:length(weights_activated)
         
        m(i,:) = weights_activated(i) * scales_activated(i,:);
    end
    m_R = - (weights_activated-1); 
    m_R_ = - (weights_activated-1); 
    miu = 1;
    miu_1 = ones(1,length(weights_activated));
    miu_2 = 1;
    for i =1:length(weights_activated)
       
        an(i,:) = m(i,:) + m_R_(i);
    end
    miu_1 = cumprod(an);
    miu_1(1:(length(weights_activated) -1),:) = [];
    miu_2 = cumprod(m_R_);
        miu = 1/ ( sum(miu_1) - (size_scales(1,2) -1)* miu_2(length(miu_2)));
    m_k =  miu .* ( miu_1 - miu_2(length(miu_2)) );
    m_H_ = miu * miu_2(length(miu_2)) ;
    gt = m_k ./ (1-m_H_);
    
end
end
%% mse_chang_pipeline_test
function [error,predict_result,gt, T1, F1, T0, F0] = mse_chang_pipeline_test(individual, input, output,  number_rules, num_scales, uscale)

size_input = size(input);
output_test = output';
error = 0;

for i = 1: size_input(1,1)  
  
    [rules_activated, weights_activated] = determine_rule_pipeline(input(i,:), individual(1: (size_input(1,2))*number_rules), individual(((size_input(1,2))*number_rules +1):(size_input(1,2) +1)*number_rules));

    scales_activated = zeros(length(rules_activated),num_scales);
    
    for j =1: length(rules_activated)
         scales_activated(j,:) = individual((size_input(1,2)+1)*number_rules +num_scales*rules_activated(j)-(num_scales-1) :(size_input(1,2)+1)*number_rules +num_scales*rules_activated(j)); 
    end
    
    gt(i,:) = return_gt(weights_activated, scales_activated);
   

    [max_val, max_idx] = max(gt(i, :)); 

   single_error(i) = uscale(max_idx);
 
end
predict_result = single_error';
num_errors = sum(predict_result ~= output);

total_samples = length(output);

error = num_errors / total_samples;

T1 = sum((predict_result == 1) & (output == 1));
F1 = sum((predict_result == 0) & (output == 1));
F0 = sum((predict_result == 1) & (output == 0));
T0 = sum((predict_result == 0) & (output == 0));
end


