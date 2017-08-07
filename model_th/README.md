# DMM - Theano code 
This folder contains theano code for building and learning Deep Markov Models.

## Files

* [`dmm.py`](dmm.py) : Main file that sets up the parameters and training/validation computational graphs
* [`evaluate.py`](evaluate.py) : File that contains code to evaluate the model (sampling/validation NLL etc.)
* [`learning.py`](learning.py) : File that implements the training loop

## Model 
In case you're unfamiliar with Theano or other computational flow graphs based frameworks 
for deep learning, the following section describes how the model is created in [`dmm.py`](dmm.py)
with pointers to the code if you're interested in modifying parts of it. 

### Parameter Estimation: A non-technical overview
The DMM comprises (1) a set of weights or parameters (think - numpy matrices) and (2) 
a procedure that specifies how the parameters are combined
to generate a sequence of simulated data. The code here sets up a computational 
flow graph that takes in data and tries to reconstruct the data using the model parameters. 

A loss function (that penalizes settings of the model parameters that do a poor job of reconstructing the data) is specified using the data
and the model's reconstruction of it. This loss function is minimized using gradient descent by updating the model parameters.  

### Parameter Estimation: More technical

* ***Step 1***: Create model parameters
    * The function `_createParams(self)`(dmm.py#29-L33) calls subfunctions `_createInferenceParams` and `_createGenerativeParams` to setup the parameters
    (of the inference network and generative model respectively) that will be estimated from data. 
    * Once the model parameters have been setup, we will use them to create the computational flow graph. 
* ***Step 2***: Setup functions to save datasets onto the GPU
    * This is done to speed up training and evaluation
* ***Step 3***: Setup Training Cost
* ***Step 4***: Setup Evaluation Functions 

## Model Parameters
The DMM has hyper-parameters specified in [`parse_args.py`](../parse_args.py). These dictate aspects of the model that control the number and sizes of 
the model parameters. Provided below is a list of parameters you can change while training the model on your data:  

