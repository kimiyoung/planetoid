# Planetoid

## Modifications

* Change implementation in python 3

## Introduction

This is an implementation of Planetoid, a graph-based semi-supervised learning method proposed in the following paper:

[Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861).
Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov.
ICML 2016.

Please cite the above paper if you use the datasets or code in this repo.

## Run the demo

We include the Citeseer dataset in the directory `data`, where the data structures needed are pickled.

To run the transductive version,

```shell
python test_trans.py
```

To run the inductive version,

```shell
python test_ind.py
```

You can refer to `test_trans.py` and `test_ind.py` for example usages of our model.

## Models

The models are implemented mainly in `trans_model.py` (transductive) and `ind_model.py` (inductive), with inheritance from `base_model.py`. You might refer to the source files for detailed API documentation.

## Prepare the data

### Transductive learning

The input to the transductive model contains:

- `x`, the feature vectors of the training instances,
- `y`, the one-hot labels of the training instances,
- `graph`, a `dict` in the format `{index: [index_of_neighbor_nodes]}`, where the neighbor nodes are organized as a list. The current version only supports binary graphs.

Let L be the number of training instances. The indices in `graph` from 0 to L - 1 must correspond to the training instances, with the same order as in `x`.

### Inductive learning

The input to the inductive model contains:

- `x`, the feature vectors of the labeled training instances,
- `y`, the one-hot labels of the labeled training instances,
- `allx`, the feature vectors of both labeled and unlabeled training instances (a superset of `x`),
- `graph`, a `dict` in the format `{index: [index_of_neighbor_nodes]}.`

Let n be the number of both labeled and unlabeled training instances. These n instances should be indexed from 0 to n - 1 in `graph` with the same order as in `allx`.

### Preprocessed datasets

Datasets for Citeseet, Cora, and Pubmed are available in the directory `data`, in a preprocessed format stored as numpy/scipy files.

The dataset for DIEL is available at <http://www.cs.cmu.edu/~lbing/data/emnlp-15-diel/emnlp-15-diel.tar.gz.> We also provide a much more succinct version of the dataset that only contains necessary files and some (not very well-organized) pre-processing code here at <http://cs.cmu.edu/~zhiliny/data/diel_data.tar.gz.>

The NELL dataset can be found here at <http://www.cs.cmu.edu/~zhiliny/data/nell_data.tar.gz.>

In addition to `x`, `y`, `allx`, and `graph` as described above, the preprocessed datasets also include:

- `tx`, the feature vectors of the test instances,
- `ty`, the one-hot labels of the test instances,
- `test.index`, the indices of test instances in `graph`, for the inductive setting,
- `ally`, the labels for instances in `allx`.

The indices of test instances in `graph` for the transductive setting are from `#x` to `#x + #tx - 1`, with the same order as in `tx`.

You can use `cPickle.load(open(filename))` to load the numpy/scipy objects `x`, `y`, `tx`, `ty`, `allx`, `ally`, and `graph`. `test.index` is stored as a text file.

## Hyper-parameter tuning

Refer to `test_ind.py` and `test_trans.py` for the definition of different hyper-parameters (passed as arguments). Hyper-parameters are tuned by randomly shuffle the training/test split (i.e., randomly shuffling the indices in `x`, `y`, `tx`, `ty`, and `graph`). For the DIEL dataset, we tune the hyper-parameters on one of the ten runs, and then keep the same hyper-parameters for all the ten runs.
