
import lasagne
from theano import sparse
import theano.tensor as T
import theano
import layers
import numpy as np
import random
import copy
from numpy import linalg as lin
from collections import defaultdict as dd

from base_model import base_model

class trans_model(base_model):
    """Planetoid-T.
    """

    def add_data(self, x, y, graph):
        """add data to the model.
        x (scipy.sparse.csr_matrix): feature vectors for training data.
        y (numpy.ndarray): one-hot label encoding for training data.
        graph (dict): the format is {index: list_of_neighbor_index}. Only supports binary graph.
        Let L and U be the number of training and dev instances.
        The training instances must be indexed from 0 to L - 1 with the same order in x and y.
        By default, our implementation assumes that the dev instances are indexed from L to L + U - 1, unless otherwise
        specified in self.predict.
        """
        self.x, self.y, self.graph = x, y, graph

    def build(self):
        """build the model. This method should be called after self.add_data.
        """
        x_sym = sparse.csr_matrix('x', dtype = 'float32')
        y_sym = T.imatrix('y')
        g_sym = T.imatrix('g')
        gy_sym = T.vector('gy')
        ind_sym = T.ivector('ind')

        l_x_in = lasagne.layers.InputLayer(shape = (None, self.x.shape[1]), input_var = x_sym)
        l_g_in = lasagne.layers.InputLayer(shape = (None, 2), input_var = g_sym)
        l_ind_in = lasagne.layers.InputLayer(shape = (None, ), input_var = ind_sym)
        l_gy_in = lasagne.layers.InputLayer(shape = (None, ), input_var = gy_sym)

        num_ver = max(self.graph.keys()) + 1
        l_emb_in = lasagne.layers.SliceLayer(l_g_in, indices = 0, axis = 1)
        l_emb_in = lasagne.layers.EmbeddingLayer(l_emb_in, input_size = num_ver, output_size = self.embedding_size)
        l_emb_out = lasagne.layers.SliceLayer(l_g_in, indices = 1, axis = 1)
        if self.neg_samp > 0:
            l_emb_out = lasagne.layers.EmbeddingLayer(l_emb_out, input_size = num_ver, output_size = self.embedding_size)

        l_emd_f = lasagne.layers.EmbeddingLayer(l_ind_in, input_size = num_ver, output_size = self.embedding_size, W = l_emb_in.W)
        l_x_hid = layers.SparseLayer(l_x_in, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        
        if self.use_feature:
            l_emd_f = layers.DenseLayer(l_emd_f, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
            l_y = lasagne.layers.ConcatLayer([l_x_hid, l_emd_f], axis = 1)
            l_y = layers.DenseLayer(l_y, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)
        else:
            l_y = layers.DenseLayer(l_emd_f, self.y.shape[1], nonlinearity = lasagne.nonlinearities.softmax)

        py_sym = lasagne.layers.get_output(l_y)
        loss = lasagne.objectives.categorical_crossentropy(py_sym, y_sym).mean()
        if self.layer_loss and self.use_feature:
            hid_sym = lasagne.layers.get_output(l_x_hid)
            loss += lasagne.objectives.categorical_crossentropy(hid_sym, y_sym).mean()
            emd_sym = lasagne.layers.get_output(l_emd_f)
            loss += lasagne.objectives.categorical_crossentropy(emd_sym, y_sym).mean()

        if self.neg_samp == 0:
            l_gy = layers.DenseLayer(l_emb_in, num_ver, nonlinearity = lasagne.nonlinearities.softmax)
            pgy_sym = lasagne.layers.get_output(l_gy)
            g_loss = lasagne.objectives.categorical_crossentropy(pgy_sym, lasagne.layers.get_output(l_emb_out)).sum()
        else:
            l_gy = lasagne.layers.ElemwiseMergeLayer([l_emb_in, l_emb_out], T.mul)
            pgy_sym = lasagne.layers.get_output(l_gy)
            g_loss = - T.log(T.nnet.sigmoid(T.sum(pgy_sym, axis = 1) * gy_sym)).sum()

        params = [l_emd_f.W, l_emd_f.b, l_x_hid.W, l_x_hid.b, l_y.W, l_y.b] if self.use_feature else [l_y.W, l_y.b]
        if self.update_emb:
            params = lasagne.layers.get_all_params(l_y)
        updates = lasagne.updates.sgd(loss, params, learning_rate = self.learning_rate)

        self.train_fn = theano.function([x_sym, y_sym, ind_sym], loss, updates = updates, on_unused_input = 'ignore')
        self.test_fn = theano.function([x_sym, ind_sym], py_sym, on_unused_input = 'ignore')
        self.l = [l_gy, l_y]

        g_params = lasagne.layers.get_all_params(l_gy, trainable = True)
        g_updates = lasagne.updates.sgd(g_loss, g_params, learning_rate = self.g_learning_rate)
        self.g_fn = theano.function([g_sym, gy_sym], g_loss, updates = g_updates, on_unused_input = 'ignore')

    def gen_train_inst(self):
        """generator for batches for classification loss.
        """
        while True:
            ind = np.array(np.random.permutation(self.x.shape[0]), dtype = np.int32)
            i = 0
            while i < ind.shape[0]:
                j = min(ind.shape[0], i + self.batch_size)
                yield self.x[ind[i: j]], self.y[ind[i: j]], ind[i: j]
                i = j

    def gen_label_graph(self):
        """generator for batches for label context loss.
        """
        labels, label2inst, not_label = [], dd(list), dd(list)
        for i in range(self.x.shape[0]):
            flag = False
            for j in range(self.y.shape[1]):
                if self.y[i, j] == 1 and not flag:
                    labels.append(j)
                    label2inst[j].append(i)
                    flag = True
                elif self.y[i, j] == 0:
                    not_label[j].append(i)

        while True:
            g, gy = [], []
            for _ in range(self.g_sample_size):
                x1 = random.randint(0, self.x.shape[0] - 1)
                label = labels[x1]
                if len(label2inst) == 1: continue
                x2 = random.choice(label2inst[label])
                g.append([x1, x2])
                gy.append(1.0)
                for _ in range(self.neg_samp):
                    g.append([x1, random.choice(not_label[label])])
                    gy.append( - 1.0)
            yield np.array(g, dtype = np.int32), np.array(gy, dtype = np.float32)

    def gen_graph(self):
        """generator for batches for graph context loss.
        """

        num_ver = max(self.graph.keys()) + 1

        while True:
            ind = np.random.permutation(num_ver)
            i = 0
            while i < ind.shape[0]:
                g, gy = [], []
                j = min(ind.shape[0], i + self.g_batch_size)
                for k in ind[i: j]:
                    if len(self.graph[k]) == 0: continue
                    path = [k]
                    for _ in range(self.path_size):
                        path.append(random.choice(self.graph[path[-1]]))
                    for l in range(len(path)):
                        for m in range(l - self.window_size, l + self.window_size + 1):
                            if m < 0 or m >= len(path): continue
                            g.append([path[l], path[m]])
                            gy.append(1.0)
                            for _ in range(self.neg_samp):
                                g.append([path[l], random.randint(0, num_ver - 1)])
                                gy.append(- 1.0)
                yield np.array(g, dtype = np.int32), np.array(gy, dtype = np.float32)
                i = j

    def init_train(self, init_iter_label, init_iter_graph):
        """pre-training of graph embeddings.
        init_iter_label (int): # iterations for optimizing label context loss.
        init_iter_graph (int): # iterations for optimizing graph context loss.
        """
        for i in range(init_iter_label):
            gx, gy = next(self.label_generator)
            loss = self.g_fn(gx, gy)
            print('iter label', i, loss) 

        for i in range(init_iter_graph):
            gx, gy = next(self.graph_generator)
            loss = self.g_fn(gx, gy)
            print('iter graph', i, loss) 

    def step_train(self, max_iter, iter_graph, iter_inst, iter_label):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        iter_graph (int): # iterations for optimizing the graph context loss.
        iter_inst (int): # iterations for optimizing the classification loss.
        iter_label (int): # iterations for optimizing the label context loss.
        """
        for _ in range(max_iter):
            for _ in range(self.comp_iter(iter_graph)):
                gx, gy = next(self.graph_generator)
                self.g_fn(gx, gy)

            for _ in range(self.comp_iter(iter_inst)):
                x, y, index = next(self.inst_generator)
                self.train_fn(x, y, index)

            for _ in range(self.comp_iter(iter_label)):
                gx, gy = next(self.label_generator)
                self.g_fn(gx, gy)

    def predict(self, tx, index = None):
        """predict the dev or test instances.
        tx (scipy.sparse.csr_matrix): feature vectors for dev instances.
        index (numpy.ndarray): indices for dev instances in the graph. By default, we use the indices from L to L + U - 1.

        returns (numpy.ndarray, #instacnes * #classes): classification probabilities for dev instances.
        """
        if index is None:
            index = np.arange(self.x.shape[0], self.x.shape[0] + tx.shape[0], dtype = np.int32)
        return self.test_fn(tx, index)


