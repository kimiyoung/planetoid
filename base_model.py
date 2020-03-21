
import lasagne
import pickle
import random
import numpy as np


class base_model(object):
    """the base model for both transductive and inductive learning."""

    def __init__(self, args):
        """
        args (an object): contains the arguments used for initalizing the model.
        """
        np.random.seed(13)
        random.seed(13)

        self.embedding_size = args.embedding_size
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.neg_samp = args.neg_samp
        self.model_file = args.model_file

        self.window_size = args.window_size
        self.path_size = args.path_size

        self.g_batch_size = args.g_batch_size
        self.g_learning_rate = args.g_learning_rate
        self.g_sample_size = args.g_sample_size

        self.use_feature = args.use_feature
        self.update_emb = args.update_emb
        self.layer_loss = args.layer_loss

        # TODO: replace lasagne
        lasagne.random.set_rng(np.random)

        self.inst_generator = self.gen_train_inst()
        self.graph_generator = self.gen_graph()
        self.label_generator = self.gen_label_graph()

    def store_params(self):
        """serialize the model parameters in self.model_file.
        """

        for i, l in enumerate(self.l):
            fout = open("{}.{}".format(self.model_file, i), 'wb')
            params = lasagne.layers.get_all_param_values(l)
            pickle.dump(params, fout, pickle.HIGHEST_PROTOCOL)
            fout.close()

    def load_params(self):
        """load the model parameters from self.model_file.
        """
        for i, l in enumerate(self.l):
            fin = open("{}.{}".format(self.model_file, i))
            params = pickle.load(fin)
            lasagne.layers.set_all_param_values(l, params)
            fin.close()

    def comp_iter(self, iter):
        """an auxiliary function used for computing the number of iterations given the argument iter.
        iter can either be an int or a float.
        """
        if iter >= 1:
            return iter
        return 1 if random.random() < iter else 0

    def train(self, init_iter_label, init_iter_graph, max_iter, iter_graph, iter_inst, iter_label):
        """training API.
        This method is a wrapper for init_train and step_train.
        Refer to init_train and step_train for more details (Cf. trans_model.py and ind_model.py).
        """
        self.init_train(init_iter_label, init_iter_graph)
        self.step_train(max_iter, iter_graph, iter_inst, iter_label)
