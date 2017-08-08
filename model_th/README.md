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

### Parameter Estimation: A non-technical overview
The DMM comprises (1) a set of weights or parameters (think - numpy matrices) and (2) 
a procedure that specifies how the parameters are combined
to generate a sequence of simulated data. The code here sets up a computational 
flow graph that takes in data and tries to reconstruct the data using the model parameters. 

A loss function (that penalizes settings of the model parameters 
that do a poor job of reconstructing the data) is specified using the data
and the model's reconstruction of it. This loss function is minimized using 
gradient descent by updating the model parameters.  

### Parameter Estimation: More technical
The DMM is a first order Gaussian state space model where the emission probabilities `p(x_t|z_t)` and 
transition probabilities `p(z_t|z_{t-1})` are parameterized by neural networks. 
We perform parameter estimation by minimizing an upper bound on the log marginal likelihood 
of the data under the model. We use recurrent neural networks (RNNs) in the inference network
to obtain a scalable method for performing probabilistic inference over the latent variables during learning. 

### Model Setup in [`dmm.py`](dmm.py)

* **Step 1** Create model parameters
    * The function [`_createParams(self)`](dmm.py#L29-L33) calls functions `_createInferenceParams` and `_createGenerativeParams` to randomly initialize the numpy matrices
    (representing parameters of the inference network and generative model respectively). 
    * With initialized model parameters, we create the computational flow graph that takes as input the raw data and returns the model's reconstruction of the data.
* **Step 2** Save dataset in memory 
    * The data is loaded onto shared variables in memory to speed up training and evaluation (the alternative being to transfer each batch of data dynamically during training time). 
    * The data `X` comprises a 3D tensor of size `Nsamples x maxT x dim_observations`. The mask `M` is a matrix of dimensions `Nsamples x maxT` where a one denotes that the time-step is observed.  
    * The dataset and the masks are stored in theano shared variables. This is done in [`_buildModel(self)`](dmm.py#L378-L401) 
    * The function `resetDataset` is used for changing the dataset dynamically at train/validation time.  
* **Step 3** Setup Training Cost
    * In [`_buildModel(self)`](dmm.py#L417-L430), we setup the computational graph used for training. 
    * The function [`_neg_elbo()`](dmm.py#L175) returns the upper bound we minimize. It has a few core components described below:
        * [`_q_z_x()`](dmm.py#L333): Given the data, this function uses RNN's to 
        estimate the parameters of the variational posterior distribution
        * [`_transition()`](dmm.py#L133): Given a tensor or matrix representing the model's latent state (dimensions `Nsamples x T x dim_stochastic` or `Nsamples x dim_stochastic`), this function computes the result of applying the transition function on the provided latent state. 
        * [`_emission()`](dmm.py#L146): Given a tensor or matrix representing the model's latent state, this function returns the result of applying the emission function on the provided latent state.  
        * [`_temporalKL()`](dmm.py#L162): This function computes the KL divergence between the variational posterior and the prior. 
* **Step 4** Setup Evaluation Functions 
    * In [`_buildModel(self)`](dmm.py#L437-L448), we define the theano functions used to evaluate the model. 

### [Learning](learning.py) and [Evaluation](evaluate.py) scripts
* [`learning.py`](learning.py) contains a simple training loop that tracks the validation loss occastionally 
* [`evaluate.py`](evaluate.py) contain helper functions to perform tasks on a model from a previous savefile 
    * `infer()` allows you to estimate the posterior distribution as fit by the inference network  
    * `reconstruct()` lets you reconstruct the time-series under the model
    * `evaluateBound()` lets you estimate the upper bound on -log p(x) on a dataset
    * `sample()` returns the result of ancestral sampling in the model

### Model Parameters
* The DMM has hyper-parameters specified in [`parse_args.py`](../parse_args.py). 
These dictate aspects of the model that control the number and sizes of the model parameters. 
* To find out more about each parameter run `python parse_args.py -h`

