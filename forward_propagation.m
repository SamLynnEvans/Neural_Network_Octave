function[a] = forward_propagation(X, theta, layer_format)

layers = length(layer_format);

a{1} = [ones(size(X, 1), 1) X]';

for i = 1:layers - 1
a{i + 1} = sigmoid(theta{i} * a{i});
if i + 1 != layers
    a{i + 1} = [ones(1, size(a{i + 1}, 2)); a{i + 1}];
end

end



end
