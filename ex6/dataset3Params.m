function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

results = eye(64,3);

% Create 64 error Rows, one for each combination of C and sigma
errorRow = 0;

% Loop through all possible values of C and sigma test
for C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
	for sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
	
	errorRow = errorRow +1;
	
	% Train the svm model using the particular C and sigma values in the loop
	
	model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
	
	% Get the predictions of the model for the cross validation set
	
	predictions = svmPredict(model,Xval);
	
	% predictions is an array of cross validation examples,
	% remember that error is defined as the fraction of the cross validation examples that were classified incorrectly...
	
	predictions_error = mean(double(predictions ~= yval))
	
	% (errorRow,:) means get the whole row
	results(errorRow,:) = [C_test, sigma_test, predictions_error];
	
	end
	
end

% Now sort the results by prediction error

sorted_results = sortrows(results,3);

% % return the best way of C

C = sorted_results(1,1);

% return the best way of sigma

sigma = sorted_results(1,2);


% =========================================================================

end
