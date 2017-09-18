"""autoencoder.py
~~~~~~~~~~~~~~

A Theano-based program for training and running an autoencoder neural network.

Supports layer types: fully connected, convolutional, max
pooling, softmax
Supports activation functions: sigmoid, tanh, and
rectified linear units

May be run on CPU and/or GPU.

Author: Janelle Lines

"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool
import json

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


'''Set the GPU constant to True or False depending on whether a GPU is being
used or not'''
#### Constants
GPU = False
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "autoencoder.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "autoencoder.py to set\nthe GPU flag to True."


'''Place the data into shared variables.  This allows Theano to copy
the data to the GPU, if one is available.'''
def shared(data):
    shared_x = theano.shared(
        np.asarray(data, dtype=theano.config.floatX), borrow=True)
    return shared_x, shared_x


#### Main class used to construct and train autoencoder
class Network(object):

    '''Initiate a neural network/autoencoder. Takes a list of `layers`,
    describing the network architecture, and
    a value for the `mini_batch_size` to be used during training
    by stochastic gradient descent.'''
    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    """Train the network using mini-batch stochastic gradient descent.
    Inputs: A set of training data, validation data, and test data. Other inputs
    include: number of epochs to be used during training, mini-batch size,
    learning rate (eta), regularization parameter (lmbda), and a monitor handel
    to declare whether or not training cost, training z-score, and validation z-score
    should be monitored over training.
    Output: Three lists that contain instances of training cost, training z-score,
    and validation z-score for each epoch.
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0,monitor = False):
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        validation_cost, validation_zscore = [], []
        training_cost, training_zscore = [], []

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 100000
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                '''if monitor:
                    training_cost.append(cost_ij)'''
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: average validation z-score {1:.5}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy <= best_validation_accuracy:
                        print("This is the best validation z-score to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test average z-score is {0:.5}'.format(
                                test_accuracy))
            if monitor:
                training_cost.append(cost_ij)
                validation_accuracy_epoch = np.mean(
                    [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                validation_zscore.append(validation_accuracy_epoch)
                train_accuracy_epoch = np.mean(
                    [train_mb_accuracy(j) for j in xrange(num_training_batches)])
                training_zscore.append(validation_accuracy_epoch)


        print("Finished training network.")
        print("Best validation average z-score of {0:.5} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test average z-score of {0:.5}".format(test_accuracy))

        return training_cost, training_zscore, validation_zscore,

    """Save the neural network to the file ``filename``.
    Input: filename in format of a string """
    def save(self, filename):
        data = {"mini_batch_size": self.mini_batch_size,
                    "parameters":[param.get_value() for param in self.params]}
        f = open(filename, "w")
        cPickle.dump(data, f,protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


    """Return the output of the network if testdata is the input.
    Input: a 31 length vector
    Output: 31 length vector output of autoencoder"""
    def feedforward(self, testdata):
        self.mini_batch_size = 1
        self.x = testdata
        testdata2 = T.stack([testdata,testdata])
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.x = T.matrix("x")
        return self.output

    """Return the output z-score of the network if testdata is input.
    Input: a 31 length vector
    Output: output z-score"""
    def zscore(self,testdata):
        x = self.feedforward(testdata)
        z = x - testdata
        return T.sum(z**2)**0.5


    """Returns the list of z ( ||y_out - y_in|| ) scores for the data set
    Input: List of length 31 lists
    Output: List of z-scores for each length 31 vector/list"""
    def list_of_zscores(self, data):
        shape = T.shape(data)
        z=[]
        for i in range(shape[0].eval()):
            z.append(self.zscore(data[i]).eval().tolist())
        return z

#### Define layer types

"""Used to create a combination of a convolutional and a max-pooling
layer."""
class ConvPoolLayer(object):

    '''Initiate a Convolutional/pool layer in neural net. Includes option to specify
    actication function'''
    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_in), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    """Method used to calculate output of layer given an input.
    Inputs: Input vector, Input vector (dropout included), mini-batch size"""
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = pool.pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers


"""Used to create a fully connected neural net layer."""
class FullyConnectedLayer(object):

    """Initiate fully connected layer
    Inputs: number of input nodes, number of output nodes, activation function,
    dropout rate"""
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_in), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]


    """Method used to calculate output of layer given an input.
    Inputs: Input vector, Input vector (dropout included), mini-batch size"""
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    """Return the cross entropy cost.
    Input: Nueral net
    Output: Cross Entropy Cost"""
    def cost(self, net):
        a = self.output_dropout
        y = net.y
        return T.sum(-y*T.log(a)-(1-y)*T.log(1-a))

    """Return the accuracy for the mini-batch.
    Input: Expected output y
    Output: Zscore """
    def accuracy(self, y):
        z = y - self.y_out
        return T.sum(z**2)**0.5

"""Used to create a softmax neural net layer."""
class SoftmaxLayer(object):

    """Initiate softmax layer
    Inputs: number of input nodes, number of output nodes,
    dropout rate"""
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    """Method used to calculate output of layer given an input.
    Inputs: Input vector, Input vector (dropout included), mini-batch size"""
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = self.output
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    """Return the log-likelihood cost.
    Input: Nueral net
    Output: Log-Likelihood Cost"""
    def cost(self, net):
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    """Return the accuracy for the mini-batch.
    Input: Expected output y
    Output: Zscore """
    def accuracy(self, y):
        z = y - self.y_out
        return T.sum(z**2)**0.5


#### Miscellanea

"""Return the size of the dataset `data`
Input: data set
Output: size of data set"""
def size(data):
    return data[0].get_value(borrow=True).shape[0]

"""Function used in dropout method"""
def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)


"""Load a neural network from the file ``filename``.  Returns an
instance of Network.
Input: filename
Output: Instance of network
"""
def load(filename):
    f = open(filename, "r")
    data = cPickle.load(f)
    f.close()
    mini_batch_size = data["mini_batch_size"]
    net = Network([
        FullyConnectedLayer(n_in=31, n_out=21,p_dropout=0.5),
        FullyConnectedLayer(n_in=21, n_out=9,p_dropout=0.5),
        FullyConnectedLayer(n_in=9, n_out=21,p_dropout=0.5),
        FullyConnectedLayer(n_in=21, n_out=31, p_dropout=0.5)],
        mini_batch_size)
    net.params = [theano.shared(param) for param in data["parameters"]]
    i=0
    for layer in net.layers:
        layer.params = [net.params[i],net.params[i+1]]
        layer.w =layer.params[0]
        layer.b = layer.params[1]
        i=i+2
    return net
