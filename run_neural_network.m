function[theta] = run_neural_network(X, y, layer_format, iter, lambda, K)

theta = create_theta(layer_format);

costFunction = @(p) nncostfunction(p, X, y, layer_format, lambda, K);

options = optimset('MaxIter', iter);

[theta, cost] = fmincg(costFunction, theta, options);

end
