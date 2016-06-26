
from scipy import sparse as sp
from ind_model import ind_model as model
import argparse
import cPickle

DATASET = 'citeseer'

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', help = 'learning rate for supervised loss', type = float, default = 0.1)
parser.add_argument('--embedding_size', help = 'embedding dimensions', type = int, default = 50)
parser.add_argument('--window_size', help = 'window size in random walk sequences', type = int, default = 3)
parser.add_argument('--path_size', help = 'length of random walk sequences', type = int, default = 10)
parser.add_argument('--batch_size', help = 'batch size for supervised loss', type = int, default = 200)
parser.add_argument('--g_batch_size', help = 'batch size for graph context loss', type = int, default = 20)
parser.add_argument('--g_sample_size', help = 'batch size for label context loss', type = int, default = 20)
parser.add_argument('--neg_samp', help = 'negative sampling rate; zero means using softmax', type = int, default = 0)
parser.add_argument('--g_learning_rate', help = 'learning rate for unsupervised loss', type = float, default = 1e-3)
parser.add_argument('--model_file', help = 'filename for saving models', type = str, default = 'ind.model')
parser.add_argument('--use_feature', help = 'whether use input features', type = bool, default = True)
parser.add_argument('--update_emb', help = 'whether update embedding when optimizing supervised loss', type = bool, default = True)
parser.add_argument('--layer_loss', help = 'whether incur loss on hidden layers', type = bool, default = True)
args = parser.parse_args()

def comp_accu(tpy, ty):
    import numpy as np
    return (np.argmax(tpy, axis = 1) == np.argmax(ty, axis = 1)).sum() * 1.0 / tpy.shape[0]

# load the data: x, y, tx, ty, allx, graph
NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
OBJECTS = []
for i in range(len(NAMES)):
    OBJECTS.append(cPickle.load(open("data/ind.{}.{}".format(DATASET, NAMES[i]))))
x, y, tx, ty, allx, graph = tuple(OBJECTS)

m = model(args)                                                 # initialize the model
m.add_data(x, y, allx, graph)                                   # add data
m.build()                                                       # build the model
m.init_train(init_iter_label = 10000, init_iter_graph = 400)    # pre-training
iter_cnt, max_accu = 0, 0
while True:
    m.step_train(max_iter = 1, iter_graph = 0.1, iter_inst = 1, iter_label = 0) # perform a training step
    tpy = m.predict(tx)                                                         # predict the dev set
    accu = comp_accu(tpy, ty)                                                   # compute the accuracy on the dev set
    print iter_cnt, accu, max_accu
    iter_cnt += 1
    if accu > max_accu:
        m.store_params()                                                        # store the model if better result is obtained
        max_accu = max(max_accu, accu)
