function[theta] = create_theta(layer_format)

layers = length(layer_format);

theta = [];
ep = 0.12;

for i = 1:layers - 1;

next_theta = rand(layer_format(i + 1) * (layer_format(i) + 1), 1) .* (2 * e) .- e;
theta = [theta; next_theta];

end
