# README #

This is software provided with the paper "Empirical Risk Minimization and Stochastic Gradient Descent for Relational Data" (http://victorveitch.com/assets/pdfs/relational_ERM.pdf)

Requires Python 3.6 with TensorFlow 1.8


# SETUP #
Run the following command in src to build the graph samplers:

python setup.py build_ext --inplace


# DEMO #
We provide a tutorial ipython notebook in notebooks/Skipgram-Demo.ipynb. It is recommended to start here.


# EXPERIMENTS #
We now describe how to reproduce the experiments in the paper. 

For the homosapiens protein-protein dataset (included):

1. Node classification---embeddings for two-stage training: 

python -m scripts.run_skipgram_simple --label-task-weight 1e-30 --sampler \<sampler\> 

where \<sampler\> is any of 
    'biased-walk': skipgram random-walk with unigram negative sampling,
    'p-sampling': p-sampling with unigram negative sampling,
    'uniform-edge': uniform edge sampling with unigram negative sampling,
    'p-sampling-induced': p-sampling with induced non-edges,
    'biased-walk-induced': induced random-walk with induced non-edges,
    'biased-walk-induced-uniform': induced random-walk with unigram negative-sampling
    'ego-open': samples open 1-neighbourhoods (not documented in paper, and tends to work poorly)

Remarks: Sampler hyperparameters for 'biased-walk' are set to match Node2Vec with a simple random walk. 
Sampler hyperparameters for other samplers are set so that the expected number of edges in a batch matches the expected number of edges in the 'biased-walk' batch (800 w/ default settings)


2. Node classification---simultaneous training:

python -m scripts.run_skipgram_simple --max-steps 120000 --label-task-weight 0.001 --global_learning_rate 10. --sampler \<sampler\> --exotic-evaluation

Remarks: the --exotic-evaluation flag causes scoring to be done averaged over all choices of samplers. Output values are most easily read in tensorboard.
Internally, the predictor uses an exponentially weighted moving average for label prediction (i.e., Polyak averaging) 


3. Wikipedia category embeddings

Get the data from http://snap.stanford.edu/data/wiki-topcats.html and extract to data/wikipedia_hlink

python -m data_processing.wikipedia_hyperlink
python -m scripts.run_wikipedia_links

Embedding visualizations were produced with TensorBoard (https://www.tensorflow.org/versions/r1.1/get_started/embedding_viz)
Category names to upload to tensorboard are included as data/wikipedia_hlink/category_names.tsv
