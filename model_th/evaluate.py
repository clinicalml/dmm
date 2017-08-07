from theano import config
from utils.misc import sampleGaussian, sampleBernoulli, unsort_idx
import numpy as np
import time
from common_eval_utils import *

"""
Functions for evaluating a DMM object
"""

def initialVals(dmm, dataset):
    mu0, cov0 = dmm.init_prior(dataset['baselines']['tensor'].astype('float32'))
    return [mu0,cov0]

def infer(dmm, dataset):
    """ Posterior Inference using recognition network 
    Returns: z,mu,logcov (each a 3D tensor) 
    Remember to multiply each by the mask of the dataset (if applicable)! 
    """
    dmm.resetDataset(dataset, quiet=True)
    return dmm.posterior_inference(idx=np.arange(dataset['features']['tensor'].shape[0]))

def reconstruct(dmm, dataset):
    dmm.resetDataset(dataset, quiet=True)
    z, _, _ = dmm.posterior_inference(idx=np.arange(dataset['features']['tensor'].shape[0]))
    bin_prob, mu, logcov = dmm.emission_fxn(z)
    return bin_prob, mu, logcov

def evaluateBound(dmm, dataset, batch_size=100):
    """ Evaluate ELBO """
    bound       = 0
    start_time  = time.time()
    N           = dataset['features']['tensor'].shape[0]
    dmm.resetDataset(dataset)
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        idx_data= np.arange(st_idx,end_idx)
        batch_bd= dmm.evaluate(idx=idx_data)
        bound  += batch_bd 
    bound /= float(dataset['features']['obs_tensor'].sum())
    end_time   = time.time()
    dmm._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds]')%(bound,end_time-start_time))
    return bound

def evaluateNLL(dmm, dataset, batch_size=20):
    """ Evaluate IS NLL"""
    nll = 0
    start_time  = time.time()
    N           = dataset['features']['tensor'].shape[0]
    dmm.resetDataset(dataset)
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        idx_data= np.arange(st_idx, end_idx)
        batch_nll= -dmm.likelihood(idx=idx_data, S = 1000)
        nll   += batch_nll 
    nll /= float(dataset['features']['obs_tensor'].sum())
    end_time   = time.time()
    dmm._p(('NLL: %.4f [Took %.4f seconds]')%(nll,end_time-start_time))
    return nll

def sample(dmm, B, U, ftypes, nsamples=100, T=20, additional = {}, stochastic = True):
    """
    Sample from Generative Model
    * We need to have baseline information about patient
    """
    assert T>1, 'Sample atleast 2 timesteps'
    #Initial sample
    mu0, cov0 = dmm.init_prior(B[:nsamples].astype('float32'))
    actions = None
    if dmm.params['dim_actions']>0:
        actions = setupU(U[:nsamples])
    if stochastic:
        z       = sampleGaussian(mu0, np.log(cov0))
    else:
        z       = mu0
    all_zs = [np.copy(z)]
    mulist  = [mu0]
    covlist = [cov0]
    for t in range(T-1):
        if dmm.params['dim_actions']==0:
            mu, cov  = dmm.transition_fxn(Z = z)
        else:
            mu, cov  = dmm.transition_fxn(Z=z, U=actions[:,[t],:])
        if stochastic:
            z        = sampleGaussian(mu,np.log(cov))
        else:
            z        = mu 
        all_zs.append(np.copy(z))
        mulist.append(np.copy(mu))
        covlist.append(np.copy(cov))
    zvec                 = np.concatenate(all_zs,axis=1)
    additional['mu_sample']     = np.concatenate(mulist, axis=1)
    additional['cov_sample']    = np.concatenate(covlist, axis=1)
    bin_prob, mu, logcov = dmm.emission_fxn(zvec)
    return bin_prob, mu, logcov, zvec

