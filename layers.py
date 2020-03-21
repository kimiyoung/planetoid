
import lasagne
from theano import sparse
import numpy as np
import theano
import theano.tensor as T

EXP_SOFTMAX = True


class DenseLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        if not EXP_SOFTMAX or self.nonlinearity != lasagne.nonlinearities.softmax:
            return self.nonlinearity(activation)
        else:
            return T.exp(activation) / (T.exp(activation).sum(1, keepdims=True))


class SparseLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(SparseLayer, self).__init__(incoming, **kwargs)

        self.num_units = num_units
        self.nonlinearity = nonlinearity

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_for(self, input, **kwargs):
        act = sparse.basic.structured_dot(input, self.W)
        if self.b is not None:
            act += self.b.dimshuffle('x', 0)
        if not EXP_SOFTMAX or self.nonlinearity != lasagne.nonlinearities.softmax:
            return self.nonlinearity(act)
        else:
            return T.exp(act) / (T.exp(act).sum(1, keepdims=True))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class HybridLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, num_units, W1=lasagne.init.GlorotUniform(), W2=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(HybridLayer, self).__init__(incomings, **kwargs)

        self.num_units = num_units
        self.nonlinearity = nonlinearity

        num_inputs_1 = self.input_shapes[0][1]
        num_inputs_2 = self.input_shapes[1][1]

        self.W1 = self.add_param(W1, (num_inputs_1, num_units), name="W1")
        self.W2 = self.add_param(W2, (num_inputs_2, num_units), name="W2")
        self.b = self.add_param(b, (num_units, ), name="b", regularizable=False)

    def get_output_for(self, inputs, **kwargs):
        act = sparse.basic.structured_dot(inputs[0], self.W1) + T.dot(inputs[1], self.W2) + self.b.dimshuffle('x', 0)
        if EXP_SOFTMAX and self.nonlinearity == lasagne.nonlinearities.softmax:
            return T.exp(act) / (T.exp(act).sum(1, keepdims=True))
        return self.nonlinearity(act)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.num_units)


class EntropyLayer(lasagne.layers.Layer):

    def __init__(self, incoming, constW, **kwargs):
        super(EntropyLayer, self).__init__(incoming, **kwargs)

        self.constW = constW

    def get_output_for(self, input, **kwargs):
        return T.reshape(T.dot(input, self.constW), (input.shape[0] * input.shape[1] * input.shape[1], 1))

    def get_output_shape_for(self, input_shape):
        if input_shape[0] is None or input_shape[1] is None:
            return (None, 1)
        return (input_shape[0] * input_shape[1] * input_shape[1], 1)


class TensorLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_units, V=lasagne.init.GlorotUniform(), W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(TensorLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.nonlinearity = nonlinearity

        num_inputs = self.input_shape[1]

        self.V = self.add_param(V, (self.num_units, num_inputs, num_inputs), name="V")
        self.W = self.add_param(W, (num_inputs, self.num_units), name="W")
        self.b = self.add_param(b, (self.num_units, ), name="b")

    def get_output_for(self, input, **kwargs):
        act = T.batched_dot(T.tensordot(input, self.V, axes=[1, 2]),
                            input) + T.dot(input, self.W) + self.b.dimshuffle('x', 0)
        return self.nonlinearity(act)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class DotLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(DotLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs[0] * inputs[1], axis=1)

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], )


class SigmoidLogLayer(lasagne.layers.Layer):

    def get_output_for(self, input, **kwargs):
        # return T.log(lasagne.nonlinearities.sigmoid(input))
        return lasagne.nonlinearities.sigmoid(input)
