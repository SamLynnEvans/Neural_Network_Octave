# Neural_Network_Octave
Neural network where users can quickly the define number of hidden layers, nodes, and output classes. 

<b>How to use</b>

The neural network is called using the following function:

run_neural_network(X, y, layer_format, iter, lambda)

The parameter layer_format contains information on the input, hidden layers, hidden nodes, and output classes. It is a vector where each row represents a layer, and the value in each row informs how many nodes there are in that layer.

Eg. If X has 400 features, the first hidden layer has 25 nodes, the second hidden layer has 20 nodes, and the output has 6 classes, layer_format would need to be assigned as such:

layer_format = [400;25;20;6];

Iter dictates how many iterations of gradient descent via the fminc function will be called. 

Lambda determines the degree of regularization.

run_neural_network returns the values of theta that are generated the user's implementation of the neural network. 

To see how these values of theta perform on your data, you can run forward_propagation(X, theta, layer_format) to return the values at each layer generated by these values of theta.

<b>Learnings</b>

After studying machine learning for two weeks (following <a href="https://www.coursera.org/learn/machine-learning">Andrew Ng's Coursera course</a>), I tried building my own simple neural network in Octave capable of solving a simple problem (see my medium post <a href="https://medium.com/@samuellynnevans/a-simple-vectorised-neural-network-in-octave-in-11-lines-of-code-b17ed9894f48">here</a>).

I then felt the perfect project to really pull together all the skills I'd developed in linear algebra, octave code, and neural network implementation would be to build a reflexive neural network able to adapt to any layer setup the user may need.

The code is entirely vectorised and this forced me to truly grabble the matrix multiplications underpinning forward and back propagation. I now understand exactly what each multiplication in these processes is doing, and why we perform them.

Overall this was a fantastic project for developing a solid base in understanding NNs and I would reccomend it to anyone else getting into ML.
