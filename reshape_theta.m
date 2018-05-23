function[theta] = reshape_theta(T, layer_format)

start = 1;
layers = length(layer_format);

for L = 1:layers - 1

nodes = layer_format(L + 1);
input = layer_format(L) + 1;
theta{L} = reshape(T(start:(start + nodes * input - 1)), nodes, input);
start = start + nodes * input;

end

end
