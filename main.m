% external reference used
% https://www.coursera.org/learn/machine-learning/home/week/3
% https://www.coursera.org/learn/machine-learning/programming/ixFof/logistic-regression
% https://www.cs.utah.edu/~piyush/teaching/8-11-print.pf

%Main entry of the program

function   main()

% Fetching  data from test and train files

% Load Training Labeled Data
T = readtable('trainLabeled.txt','Format','%f%f%f%s');
Train_Labeled_table = T(:,1:4);
XTrainLabel = T(:,1:3);

%Converting Training data to array format
XTrainLabel= table2array(XTrainLabel);
yTrainLabel = T(:,4);

%converting Training label to array format
ytrLabel = table2array(yTrainLabel);

%creating 1d arary with zeros value
ytrain = zeros(length(ytrLabel),1);

N = length(ytrain);


%loading Training Unlabled Data
TUnLabeled = readtable('trainUnLabeled.txt','Format','%f%f%f');
Train_UnLabeled_table = TUnLabeled(:,1:3);
XTrainUnLabel = TUnLabeled(:,1:3);

%Converting Training data to array format
XTrainUnLabel= table2array(XTrainUnLabel);
yTrainUnLabel= zeros(length(XTrainUnLabel),1);

%loading test data
TestTable = readtable('test.txt','Format','%f%f%f%s');
XTest = TestTable(:,1:3);
XTest = table2array(XTest);
yTestLabel = TestTable(:,4);

%converting Training label to array format
ytestLabel = table2array(yTestLabel);

%creating 1d arary with zeros value
ytest = zeros(length(ytestLabel),1);

%end loading  data

%preprocessing data

%converting training labels to 1 and 0 and stroing it in the variable
%ytrain

for i=1:length(ytrLabel)
    j = char(T{i,4});
    if(j == 'W')
        ytrain(i,:) = 0;
    else
        ytrain(i,:) = 1;
    end
end

%converting test labels to 1 and 0 and stroing it in the variable
%ytest
for i=1:length(ytest)
    j = char(TestTable{i,4});
    if(j == 'W')
        ytest(i,:) = 0;
    else
        ytest(i,:) = 1;
    end
end


% %preprocessing data

% Preparing data matrix using Training dataset
[m, n] = size(XTrainLabel);

% Add intercept term to training datasets X
XTrainLabel = [ones(m, 1) XTrainLabel];

% Add intercept term to unlabeld training datasets X
XTrainUnLabel = [ones(length(XTrainUnLabel), 1) XTrainUnLabel];


% Add intercept term to test set X_test
XTest = [ones(length(XTest), 1) XTest];


% 
%setting learning rate and no of iterations for individual logistic
%regression classifier
alpha = 0.01;
noofiterations = 1200;
initial_theta = randi([0, 1],n+1,1);

%matrix to store theta values per iterations
theta_values = zeros(length(initial_theta),m);
% 
%performing classification using single logistic regression

[theta_single_classifier] = gradientDescent(XTrainLabel, ytrain, initial_theta, alpha, noofiterations);
result_test = 0;
accuracy_test = 0;
error_count = 0;

XT = 5;
while XT > 0
     u = length(XTrainLabel);
     confidenceSample = zeros(length(XTrainUnLabel),1);
     yUnlabeld = zeros(length(XTrainUnLabel),1);
     xUnlabeled = zeros(length(XTrainUnLabel),4);
     indexOfXElem = zeros(length(XTrainUnLabel),1);
     %Applying learned theta parameters in unlabeled data and storing
     %confidence value based on h(sigmoid(theta*x)) value
     for k = 1:length(XTrainUnLabel)
       prob = sigmoid(XTrainUnLabel(k,:) * theta_single_classifier);
       xUnlabeled(k,:) = XTrainUnLabel(k,:);
       indexOfXElem(k,:) = k;
       if round(prob) == 1
           confidenceSample(k,:) = prob;
           yUnlabeld(k,:) = 1;
       else
           confidenceSample(k,:) = 1- prob;
           yUnlabeld(k,:) = 0;
       end       
     end

     %get k highest confidence items and store the corresponding datapoint
     %from unlabled dataset to labeled dataset and remove the datapoint
     %from unlabled dataset.
     
     count = 5;
     while count > 0
         u = u + 1;
         [elem,indt] = max(confidenceSample);
         XTrainLabel(u,:) = XTrainUnLabel(indt,:);
         ytrain(u,:) = yUnlabeld(indt);
         XTrainUnLabel(indt,:) = [];
         yUnlabeld(indt,:) = [];
         confidenceSample(indt,:) = [];
         count = count -1;
     end
    
    XT = XT -1;
    
end
 
% Predict output label for a test data set using single logistic
% regression classifier

for k = 1:length(XTest)
    prob = sigmoid(XTest(k,:) * theta_single_classifier);
    if prob >=0.5
        result_test = 1;
    else
        result_test = 0;
    end
    if result_test == ytest(k)
        accuracy_test = accuracy_test + 1;
    end
    fprintf('predicted class label is %d and true class label was %d\n',result_test,ytest(k));
end

fprintf('\n');
error_count = length(XTest)- accuracy_test;
fprintf('Error rate and Accuracy for single logistic regression classifier on test data %d%%,%d%%\n',round((error_count/length(ytest))*100),round((accuracy_test/length(ytest))*100));



fprintf('\n');




%learning theta parameters in the merged dataset(both labeled and unlabaled
%dataset)
new_m = length(XTrainLabel);
theta_values_merge = zeros(length(initial_theta),new_m);


%performing classification using single logistic regression
[theta_single_classifier_new] = gradientDescent(XTrainLabel, ytrain, initial_theta, alpha, noofiterations);

    
combined_result_test = 0;
combined_accuracy_test = 0;
combined_error_count = 0;

for k = 1:length(XTest)
    prob = sigmoid(XTest(k,:) * theta_single_classifier_new);
    if prob >=0.5
        combined_result_test = 1;
    else
        combined_result_test = 0;
    end
    if combined_result_test == ytest(k)
        combined_accuracy_test = combined_accuracy_test + 1;
    end
    fprintf('predicted class label is %d and true class label was %d\n',combined_result_test,ytest(k));
end

fprintf('\n');
combined_error_count = length(XTest)- combined_accuracy_test;
fprintf('Error rate and Accuracy for logistic regression classifier using semi supervised learning algorithm on test data %d%%,%d%%\n',round((combined_error_count/length(ytest))*100),round((combined_accuracy_test/length(ytest))*100));

%end of programme

end

