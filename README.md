# Deep Markov Models

## Overview 

This repository contains cleaner and more modular theano code for 
implementing Deep Markov Models. 

The code is documented and should be easy to modify. 

## Requirements
This code has the following requirements:
* `python 2.7` 
* `theano`
* `theanomodels`

## Folders 
* [`model_th`](model_th/): This folder contains raw theano code implementing the model. See the folder for details on how the DMM was implementation
and pointers to portions of the code. 
* [`dmm_data`](dmm_data/): This folder contains code to load the polyphonic music data and a synthetic dataset. 
* [`ipynb`](ipynb/): This folder contains some IPython notebooks with examples on loading and running the model on your own data.  
* [`parse_args.py`](parse_args.py): This file contains hyperparameters used by the model. See the [`README.md`](model/README.md) for an explanation of 
what the various choices of parameters change in the generative model and inference network.  
* [`expt`](expt/): Setup for running the DMM on polyphonic music. 

## Parameter Estimation in DMMs

## Research Code

## Paper

Please cite the following paper if you find the code useful in your research: 
