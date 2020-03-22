import numpy as np


# part 1
# Implement the following functions, which are used to carry out the forward propagation process:


def initialize_parameters(layer_dims):
    """
    :param layer_dims: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened input, layer L is the output softmax) 
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL). 
    """

    parameters = [None]
    for layer in range(1, len(layer_dims)):
        W = np.random.rand(layer_dims[layer], layer_dims[layer - 1]) + 0.5  # values 0.5 - 1.5
        b = np.zeros(layer_dims[layer])
        layer_dict = {'W': W, 'b': b}
        parameters.append(layer_dict)
    return parameters


def linear_forward(A, W, B):
    """
    Implement the linear part of a layer's forward propagation.
    :param A: the activations of the previous layer.
    :param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer]).
    :param B: the bias vector of the current layer (of shape [size of current layer, 1]).
    :return:
     Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
     linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute) 
    """

    # Z = W * A + B
    Z = np.matmul(W,A) + B
    linear_cache = {'A': A, 'W': W, 'B': B}
    return Z, linear_cache


def softmax(Z):
    """
    :param Z: the linear component of the activation function
    :return: A – the activations of the layer, activation_cache – returns Z, which will be useful for the backpropagation
    """
    # second version to compute softmax
    # exp = np.exp(Z - np.max(Z))
    # A = exp / exp.sum(axis=0)
    exp_scores = np.exp(Z)
    A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    activation_cache = Z
    return A, activation_cache


def relu(Z):
    """
    :param Z: the linear component of the activation function
    :return: A – the activations of the layer, activation_cache – returns Z, which will be useful for the backpropagation
    """

    A = np.maximum(Z, 0)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations of the previous layer.
    :param W: the weights matrix of the current layer.
    :param B: the bias vector of the current layer.
    :param activation: the activation function to be used (a string, either “softmax” or “relu”).
    :return:
    A – the activations of the current layer.
    cache – a joint dictionary containing both linear_cache and activation_cache.
    """

    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation is "softmax":
        A, activation_cache = softmax(Z)
    elif activation is "relu":
        A, activation_cache = relu(Z)
    else:
        raise Exception("activation should be either softmax or relu")
    cache = {'linear_cache': linear_cache, 'activation_cache': activation_cache}
    return A, cache


def L_model_forward(X, parameters, use_batchnorm):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation.
    :param X: the data, numpy array of shape (input size, number of examples).
    :param parameters: the initialized W and b parameters of each layer.
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation.
    :return:
    AL – the last post-activation value.
    caches – a list of all the cache objects generated by the linear_forward function.
    """

    caches = []
    first_layer = 1
    last_layer = len(parameters)-1

    # first layer
    A, cache = linear_activation_forward(X, parameters[first_layer]['W'], parameters[first_layer]['B'], "relu")
    caches.append(cache)
    if use_batchnorm:
        A = apply_batchnorm(A)
    # hidden layers
    for layer in range(2, last_layer):
        A, cache = linear_activation_forward(A, parameters[layer]['W'], parameters[layer]['B'], "relu")
        caches.append(cache)
        if use_batchnorm:
            A = apply_batchnorm(A)
    # last layer
    AL, cache = linear_activation_forward(A, parameters[last_layer]['W'], parameters[last_layer]['B'], "softmax")
    caches.append(cache)
    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss.
    :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples).
    :param Y: the labels vector (i.e. the ground truth).
    :return: the cross-entropy cost
    """
    data_loss = 0
    m = len(AL[0])
    for r in range(m):  # For each element in the batch
        for c in range(len(Y[r, :])):  # For each class
            if Y[r, c] != 0:  # Positive classes
                data_loss += -np.log(AL[r, c]) * Y[r, c]  # We sum the loss per class for each element of the batch
    data_loss = -1 / m * data_loss
    return data_loss


def apply_batchnorm(A):
    """
    performs batchnorm on the received activation values of a given layer.
    :param A: the activation values of a given layer.
    :return: the normalized activation values, based on the formula learned in class.
    """
    mean = np.mean(A, axis=0)
    variance = np.mean((A - mean) ** 2, axis=0)
    float_epsilon = np.finfo(float).eps
    NA = (A - mean) * 1.0 / np.sqrt(variance + float_epsilon)
    return NA


# part 2
# Implement the following functions, which are used to carry out the backward propagation process:

def Linear_backward(dZ, cache):
    """
    description: Implements the linear part of the backward propagation process for a single layer  

    Input:
    dZ – the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Output:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    pass


def linear_activation_backward(dA, cache, activation):
    """
    Description:
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.

    Some comments:
    •	The derivative of ReLU is
    •	The derivative of the softmax function is: , where  is the softmax-adjusted probability of the class and  is the “ground truth” (i.e. 1 for the real class, 0 for all others)
    •	You should use the activations cache created earlier for the calculation of the activation derivative and the linear cache should be fed to the linear_backward function 

    Input:
    dA – post activation gradient of the current layer
    cache – contains both the linear cache and the activations cache

    Output:
    dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    pass


def relu_backward(dA, activation_cache):
    """
    Description:
    Implements backward propagation for a ReLU unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z 
    """
    pass


def softmax_backward(dA, activation_cache):
    """
    Description:
    Implements backward propagation for a softmax unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z 
    """
    pass


def L_model_backward(AL, Y, caches):
    """
    Description:
    Implement the backward propagation process for the entire network.

    Some comments:
    the backpropagation for the softmax function should be done only once as only the output layers uses it and the RELU should be done iteratively over all the remaining layers of the network.

    Input:
    AL - the probabilities vector, the output of the forward propagation (L_model_forward)
    Y - the true labels vector (the "ground truth" - true classifications)
    Caches - list of caches containing for each layer: a) the linear cache; b) the activation cache

    Output:
    Grads - a dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    pass


def Update_parameters(parameters, grads, learning_rate):
    """
    Description:
    Updates parameters using gradient descent

    Input:
    parameters – a python dictionary containing the DNN architecture’s parameters
    grads – a python dictionary containing the gradients (generated by L_model_backward)
    learning_rate – the learning rate used to update the parameters (the “alpha”)

    Output:
    parameters – the updated values of the parameters object provided as input
    """
    pass


# part 3
# In this section you will use the functions you created in the previous sections to train the network and produce predictions.

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Description:
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function, and the final layer will apply the softmax activation function. The size of the output layer should be equal to the number of labels in the data. Please select a batch size that enables your code to run well (i.e. no memory overflows while still running relatively fast).

    Hint: the function should use the earlier functions in the following order: initialize -> L_model_forward -> compute_cost -> L_model_backward -> update parameters

    Input:
    X – the input data, a numpy array of shape (height*width , number_of_examples)
    Comment: since the input is in grayscale we only have height and width, otherwise it would have been height*width*3
    Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    Layer_dims – a list containing the dimensions of each layer, including the input batch_size – the number of examples in a single training batch.

    Output:
    parameters – the parameters learnt by the system during the training (the same parameters that were updated in the update_parameters function).
    costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values). 
    """
    pass


def Predict(X, Y, parameters):
    """
    Description:
    The function receives an input data and the true labels and calculates the accuracy of the trained neural network on the data.

    Input:
    X – the input data, a numpy array of shape (height*width, number_of_examples)
    Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    Parameters – a python dictionary containing the DNN architecture’s parameters 

    Output:
    accuracy – the accuracy measure of the neural net on the provided data (i.e. the percentage of the samples for which the correct label receives the hughest confidence score). Use the softmax function to normalize the output values.
    """
    pass


# Tests
# layer_dims_test = [3, 4, 1]
# d = initialize_parameters(layer_dims_test)
# a = np.array([[1,2], [3,4]])
# a = a.flatten()
# x = np.ones((3,1)) + 0.5
# w = np.ones((4,3)) + 1
# z = np.matmul(w,x)
# AL = np.array([[0, 0], [0, 0], [0, 0]])
# Y = np.array([[1, 0], [0, 1], [0, 0]])
# cost = compute_cost(AL, Y)
# print(cost)
# a = np.random.rand(3,2)
a = np.array([[1,2], [3,4]])
na = apply_batchnorm(a)
print("done")
