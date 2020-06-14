function [] = linearRegression(data)
% Linear regression calculus for Yahoo Finance 
% historical stock prices in csv format.

%
% functions
%

function [price] = get_price(date,theta)
    date = datenum(date);
    price = [1, date] * theta;
endfunction

function retval = hypotesis (theta, X, y)
  retval = (theta(1,1) + (theta(2,1) * X(:,2,1) - y));
endfunction

function J = computeCost(X, y, theta)
    %COMPUTECOST Compute cost for linear regression
    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y

    m = length(y);
    predictions = X * theta;
    sqrErrors = (predictions - y).^2;

    J = 1/(2*m) * sum(sqrErrors);
endfunction

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

    m = length(y);
    J_history = zeros(num_iters, 1);    

    for iter = 1:num_iters
        temp0 = theta(1,1) - (alpha/m) * (sum(hypotesis(theta, X, y)));
        temp1 = theta(2,1) - (alpha/m) * (sum(hypotesis(theta, X, y).*X(:,2,1)));
        theta(1,1) = temp0;
        theta(2,1) = temp1;

        J_history(iter) = computeCost(X, y, theta);
    end
endfunction

%
% main
%

date = importdata(data);

% Delete the headers
date.textdata(1) = [];

y = date.data(:,1);
textdata = cell2mat(date.textdata);

for i= 1:length(textdata)
    aux = textdata(i,:);
    X(i,1) = datenum(sprintf('%s/%s',cell2mat(strsplit(aux,'-')(2)),cell2mat(strsplit(aux,'-')(3))));
end

plot(X,y,'rx','MarkerSize',20);
ylabel('Last Price');
xlabel('Date');
hold on

m = length(y);
X = [ones(m, 1), X(:,1)];
theta = zeros(2, 1);

iterations = 100
alpha = 0.000000000001

t_ = gradientDescent(X, y, theta, alpha, iterations);
plot(X(:,2), X*t_, '-');

fprintf('\nPrice in 12/31: U$%d\n', get_price("12/31",t_));

end
