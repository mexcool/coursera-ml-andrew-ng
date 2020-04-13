function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X_temp = [ones(size(X,1), 1) X];
model = X * theta;
J = 1/2/m * sum((model - y).^2) + lambda/2/m * sum(theta(2:end,1).^2);

theta_no_bias = theta;
theta_no_bias(1) = 0;
grad = 1/m * ((model - y)' * X)' + lambda/m * theta_no_bias;

% grad1 = 1/m * ((model - y)' * X)';
% grad = grad1 + lambda/m * theta_no_bias;


% =========================================================================

grad = grad(:);

end
