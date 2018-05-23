function[J grad] = nncostfunction(T, X, y, layer_format, lambda, K)

m = length(y);

layers = length(layer_format);

theta = reshape_theta(T, layer_format);

%forward_prop

a = forward_propagation(X, theta, layer_format);

%format y

c = 1:K;

y = y == c;

%cost function

regularisation = 0;

for L = 1:layers - 1

regularisation = regularisation + sum(sum(theta{L}(:, 2:end).^2)); 

end

regularisation = regularisation * lambda/(2 * m);

J = 1/m * sum(diag(-y * log(a{layers}) - (1 - y) * log(1 - a{layers}))) + regularisation;

%back propagation

sigma{layers} = a{layers} - y';

for L = layers - 1:-1:2
   
sigma{L} = ((theta{L}' * sigma{L + 1}) .* (a{L}.*(1-a{L})))(2:end, :);

end

%calculate delta

grad = [];

for L = 1:layers-1

Delta{L} = sigma{L + 1} * a{L}';
Delta{L} = 1/m .* Delta{L} + (lambda/m)*[zeros(size(theta{L},1), 1) theta{L}(:, 2:end)];
grad = [grad; Delta{L}(:)];

end

end
