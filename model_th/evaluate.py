from theano import config
from utils.misc import sampleGaussian, sampleBernoulli, unsort_idx
import numpy as np
import time

"""
Functions for evaluating a DMM object
"""
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

def sampleGaussian(mu,logcov):
        return mu + np.random.randn(*mu.shape)*np.exp(0.5*logcov)

def sample(dmm, dataset, nsamples=100, T=10, additional = {}):
    """
    Sample from Generative Model
    * We need to have baseline information about patient
    """
    assert T>1, 'Sample atleast 2 timesteps'
    #Initial sample
    mu0, cov0 = dmm.init_prior(dataset['baselines']['tensor'][:nsamples].astype('float32'))
    if dmm.params['dim_actions']>0:
        actions = dataset['actions']['tensor'][:nsamples]
    else:
        actions = None
    z         = sampleGaussian(mu0, np.log(cov0))
    all_zs = [np.copy(z)]
    additional['mu']     = []
    additional['cov'] = []
    for t in range(T-1):
        if dmm.params['dim_actions']==0:
            mu, cov  = dmm.transition_fxn(z)
        else:
            mu, cov  = dmm.transition_fxn(z, U=actions[:,[t],:])
        z            = sampleGaussian(mu,np.log(cov))
        all_zs.append(np.copy(z))
        additional['mu'].append(np.copy(mu))
        additional['cov'].append(np.copy(cov))
    zvec                 = np.concatenate(all_zs,axis=1)
    additional['mu']     = np.concatenate(additional['mu'], axis=1)
    additional['cov'] = np.concatenate(additional['cov'], axis=1)
    bin_prob, mu, logcov = dmm.emission_fxn(zvec)
    return bin_prob, mu, logcov, zvec
