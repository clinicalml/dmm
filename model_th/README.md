# DMM - Theano code 
This folder contains theano code for building and learning Deep Markov Models.

## Files

* [`dmm.py`](dmm.py) : Main file that sets up the parameters and training/validation computational graphs
* [`evaluate.py`](evaluate.py) : File that contains code to evaluate the model (sampling/validation NLL etc.)
* [`learning.py`](learning.py) : File that implements the training loop

## Model 
If you are unfamiliar with Theano or other computational flow graphs based frameworks 
for deep learning, the following section describes how the model is created in [`dmm.py`](dmm.py)
with pointers to the code if you're interested in modifying parts of it. 

### Parameter Estimation: Non-technical overview
The DMM comprises (1) a set of weights or parameters (think - numpy matrices) and (2) 
a procedure that specifies how the parameters are combined
to generate a sequence of simulated data. The code here sets up a computational 
flow graph that takes in data and tries to reconstruct the data using the model parameters. 
A loss function (that penalizes settings of the model parameters 
that do a poor job of reconstructing the data) is specified using the data
and the model's reconstruction of it. This loss function is minimized using 
gradient descent by updating the model parameters.  

### Parameter Estimation: Technical
* The DMM is a first order Gaussian state space model where the emission `p(x_t|z_t)` and transition probabilities `p(z_t|z_{t-1})` are parameterized by neural networks. 
* Since this is a latent variable model (`z1...zT` are latent), we use variational inference for Maximum Likelihood learning of model parameters by minimizing an upper bound on the log marginal likelihood of the data under the model. 
* To perform variational inference at train and test time (`q(z1...zT|x1....xT)`), we designed an inference network 
(a regression function from the raw data to the variational parameters) using recurrent neural networks (RNNs).
* The structure of the inference network (hence the name `Structured Inference Networks`) is derived from the factorization of the true posterior distribution of the generative model. We refer the reader to the [paper](https://arxiv.org/abs/1609.09869) for more details. 

### Model Setup
To create the model in Theano, we follow the steps detailed below:

* **Step 1** Create the model parameters
    * The function [`_createParams(self)`](dmm.py#L29-L128) calls functions `_createInferenceParams` and `_createGenerativeParams` to randomly initialize the numpy matrices
    (representing parameters of the inference network and generative model respectively). 
    * With initialized model parameters, we can create the computational flow graph that takes as input the raw data and returns the model's reconstruction of the data.
* **Step 2** Save dataset in memory 
    * The dataset and the masks (to denote whether an observation `x_t` is valid for different choices of `t`) are stored 
    in theano shared variables. This is done in [`_buildModel(self)`](dmm.py#L375-L398) 
    * The data is loaded onto [shared variables](dmm.py#L386-L387) in memory to speed up training and evaluation. 
    * The function `resetDataset()` can be used for changing the dataset dynamically at train/validation time. 
    * The data is subsampled using a vector of indices `idx`. This is used to slice the data yielding `X` a 3D tensor of size `Nsamples x maxT x dim_observations` and a mask `M` -- a matrix of dimensions `Nsamples x maxT` where a one denotes that the time-step is observed.  
* **Step 3** Setup Training Cost
    * In [`_buildModel(self)`](dmm.py#L414-L428), we setup the computational graph used for training. 
    * The function [`_neg_elbo()`](dmm.py#L181-L200) returns the upper bound we minimize. It has a few core components described below:
        * [`_q_z_x()`](dmm.py#L335-L352): Given the data `X`, this function uses RNN's to approximate the variational parameters for `q(z1...zT|x1...xT)`. 
        * [`_transition()`](dmm.py#L131-L150): Used to compute `p(z_t|z_{t-1})`. Given a tensor or matrix representing the model's latent state (dimensions `Nsamples x T x dim_stochastic` or `Nsamples x dim_stochastic`) `z_t`, this function computes the result of applying the transition function to it. 
        * [`_emission()`](dmm.py#L152-L162): Used to compute `p(x_t|z_t)`. Given a tensor or matrix representing the model's latent state `z_t`, this function returns an intermediate hidden state used to compute the distributional parameters of the data resulting from applying the emission function on the provided latent state.
        * [`_temporalKL()`](dmm.py#L168): This function computes the KL divergence between the variational posterior and the prior.
* **Step 4** Setup Evaluation Functions 
    * In [`_buildModel(self)`](dmm.py#L431-L445), we define the theano functions used to evaluate the model. 

### [Learning](learning.py) and [Evaluation](evaluate.py) scripts
* [`learning.py`](learning.py) contains a training loop that iterates over batches of the training data and performs gradient descent in the model parameters. 
There is also code to track the validation loss. Modify this file to add other statistics you wish to track. 
* [`evaluate.py`](evaluate.py) contain helper functions to perform tasks on a model from a previous savefile 
    * `infer()` allows you to estimate the posterior distribution as fit by the inference network 
    * `reconstruct()` lets you reconstruct the time-series under the model
    * `evaluateBound()` lets you estimate the upper bound on -log p(x) on held-out data
    * `sample()` returns the result of ancestral sampling in the model

### Hyperparameters 
* The dimensionality of the latent variables `z1...zT`, the number of layers in the multi-layer perceptrons (MLPs) that comprise the emission and transition function are among the hyperparameters of the model. 
* These hyper-parameters are specified in [`parse_args.py`](../parse_args.py). 
* To find out more about each parameter run `python parse_args.py -h`
