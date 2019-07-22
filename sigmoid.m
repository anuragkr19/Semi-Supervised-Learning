function g = sigmoid(z)
%   g = SIGMOID(z) computes the sigmoid of z.

%Intialize scaler g
g = zeros(size(z));
g = 1.0./(1.0 + exp(-z));

end
