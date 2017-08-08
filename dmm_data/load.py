from datasets.load import loadDataset 
import os
import numpy as np
from utils.misc import readPickle, savePickle

def simulateLinearData(N, T, DIM):
    """ Synthetic data generated according to a first order linear Markov process """
    z    = np.random.randn(N, DIM)
    zlist= [np.copy(z)[:,None,:]]
    W    = 0.1*np.random.randn(DIM,DIM)
    for t in range(T-1):
        z_next = np.dot(z,W) 
        zlist.append(np.copy(z_next)[:,None,:])
        z      = z_next
    Z   = np.concatenate(zlist, axis=1)
    X   = Z + 4*np.random.randn(*Z.shape) 
    return X, Z

def loadSyntheticData():
    curdir = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(curdir+'/synthetic.pkl'): 
        print 'Reloading...'
        return readPickle(curdir+'/synthetic.pkl')[0]
    """ Generate simple synthetic data """
    params  = {}
    params['train']  = 10000
    params['valid']  = 1000
    params['test']   = 1000
    N       = np.sum([params[k] for k in params])
    T       = 10
    DIM_OBS = 3
    np.random.seed(0)
    data, data_Z     = simulateLinearData(N, T, DIM_OBS)
    """
    Split into train/valid/test
    """
    shufidx = np.random.permutation(N)
    indices = {}
    indices['train'] = shufidx[:params['train']]
    indices['valid'] = shufidx[params['train']:params['train']+params['valid']]
    indices['test']  = shufidx[params['train']+params['valid']:]
    """
    Setup dataset to return
    """
    dataset = {}
    for k in ['train','valid','test']:
        dataset[k]   = {}
        dataset[k]['tensor']   = data[indices[k]] 
        dataset[k]['tensor_Z'] = data_Z[indices[k]] 
        dataset[k]['mask']     = np.ones_like(dataset[k]['tensor'][:,:,0]) 
    dataset['data_type']            = 'real'
    dataset['dim_observations']     = 3
    savePickle([dataset],curdir+'/synthetic.pkl')
    print 'Saving...'
    return dataset

def load(dset):
    if dset   in ['jsb','nottingham','musedata','piano']:
        musicdata = loadDataset(dset)
        dataset   = {}
        for k in ['train','valid','test']:
            dataset[k] = {}  
            dataset[k]['tensor'] = musicdata[k] 
            dataset[k]['mask']   = musicdata['mask_'+k]
        dataset['data_type']        = musicdata['data_type']
        dataset['dim_observations'] = musicdata['dim_observations']
    elif dset == 'synthetic':
        dataset = loadSyntheticData()
    else:
        raise ValueError('Invalid dataset: '+dset)
    return dataset

if __name__=='__main__':
    #data = load('jsb')
    data = load('synthetic')
    import ipdb; ipdb.set_trace()
