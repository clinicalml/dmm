from theano import config
from utils.misc import sampleGaussian, sampleBernoulli, unsort_idx
import numpy as np
import time

"""
Functions for evaluating DMMs
"""
def infer(dmm, dataset):
    """ Posterior Inference using recognition network 
    Returns: z,mu,logcov (each a 3D tensor) 
    """
    dmm.resetDataset(dataset, quiet=True)
    return dmm.posterior_inference(idx=np.arange(dataset['tensor'].shape[0]))

def reconstruct(dmm, dataset):
    dmm.resetDataset(dataset, quiet=True)
    z, _, _ = dmm.posterior_inference(idx=np.arange(dataset['tensor'].shape[0]))
    params  = dmm.emission_fxn(z)
    if dmm.params['data_type'] == 'real':
        return params[0], params[1]
    elif dmm.params['data_type']=='binary':
        return params

def evaluateBound(dmm, dataset, batch_size=100):
    """ Evaluate ELBO """
    bound       = 0
    start_time  = time.time()
    N           = dataset['tensor'].shape[0]
    dmm.resetDataset(dataset)
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        idx_data= np.arange(st_idx,end_idx)
        batch_bd= dmm.evaluate(idx=idx_data)
        bound  += batch_bd 
    bound /= float(dataset['mask'].sum())
    end_time   = time.time()
    dmm._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds]')%(bound,end_time-start_time))
    return bound

def sample(dmm, nsamples=100, T=20, additional = {}, stochastic = True):
    """ Sample from Generative Model """
    assert T>1, 'Sample atleast 2 timesteps'
    mu0       = np.zeros((nsamples, 1, dmm.params['dim_stochastic']))
    cov0      = np.ones_like(mu0)
    if stochastic:
        z       = sampleGaussian(mu0, np.log(cov0))
    else:
        z       = mu0
    all_zs  = [np.copy(z)]
    mulist  = [mu0]
    covlist = [cov0]
    for t in range(T-1):
        mu, cov  = dmm.transition_fxn(Z = z)
        if stochastic:
            z        = sampleGaussian(mu,np.log(cov))
        else:
            z        = mu 
        all_zs.append(np.copy(z))
        mulist.append(np.copy(mu))
        covlist.append(np.copy(cov))
    zvec                        = np.concatenate(all_zs,axis=1)
    additional['mu_sample']     = np.concatenate(mulist, axis=1)
    additional['cov_sample']    = np.concatenate(covlist, axis=1)
    params     = dmm.emission_fxn(zvec)
    if type(params) is not list:
        params = [params]
    return params, zvec 
