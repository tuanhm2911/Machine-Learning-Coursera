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
grad = zeros(size(theta));                   % grad: 2x1

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

diff = X * theta - y;                               % X: 21x2; theta: 2x1
theta_reg = [0; theta(2:end,:)];
J = sum(diff .^ 2)/(2*m) + lambda/(2*m)*sum(sum(theta_reg .^ 2));

grad = (X'*diff + lambda*theta_reg)/m;
grad = grad(:);

end
