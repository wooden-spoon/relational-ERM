# README #
 
This is a demo for model development using relational ERM and the associated tensorflow integration. 
We consider data consisting of a graph where each node has a class label and a real-vector valued feature.
The model uses node embedding vectors to predict graph structure (with a standard skipgram approach) and 
predicts class identities by using a neural network, taking the embeddings and the node features as input.

Critically, this demo illustrates the use of batch updates using relational ERM models. This can dramatically speed up training.

# PRELIMINARIES #

Follow the setup instructions in the top-level readme (i.e., install tensorflow and compile the samplers). 
It's also recommended to go through the ipython notebook tutorial first.

To run the code on the reddit data from graphsage (http://snap.stanford.edu/graphsage/)
1. download the reddit data from http://snap.stanford.edu/graphsage/ and extract to `data/raw_data`
2. from src, run `python -m node_classification_with_features.dataset_logic.graph_sage_preprocess --data-dir ../data/raw_data`
3. Save the output `reddit.npz` to `data/reddit/reddit.npz` 

# EXECUTION #

From src, run:
`python -m node_classification_with_features.run_classifier --train-dir ../logging/reddit --data-dir ../data/reddit`

Tensorboard logging files are stored in train-dir (logging/reddit in the example). The trained model can be inspected using tensorboard, following the ordinary tensorflow workflow.

# STRUCTURE #

`sample.py` : high level logic for constructing SAMPLE using the provided tools
`node_classification_template.py` : template for constructing tensorflow estimators for node classification on graphs using vector valued features
`predictor_class_and_losses.py` : Specifies a (neural-net based) predictor class and associated loss function. Constructs an estimator by calling `node_classification_template.py`
`run_classifier.py` : combines the model ingredients, runs training, and runs evaluation

