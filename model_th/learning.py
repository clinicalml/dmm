import evaluate as DMM_evaluate
import numpy as np
from utils.misc import saveHDF5
import time

""" Functions for learning a DMM object """
def learn(dmm, dataset, epoch_start=0, epoch_end=1000, 
          batch_size=200, shuffle=True, savefreq=None, savefile = None, 
          dataset_eval = None):
    """ Train a DMM using data
    dmm: DMM object
    dataset, dataset_eval: <dict> object with data
    epoch_start, epoch_end, batch_size, savefreq: <int> 
    savefile: <str> Savefile for intermediate results
    """
    assert not dmm.params['validate_only'],'cannot run function in validate only mode'
    N         = dataset['tensor'].shape[0]
    idxlist   = range(N)
    batchlist = np.split(idxlist, range(batch_size,N,batch_size))
    bound_train_list, bound_valid_list = [],[]
    epfreq = 1
    dmm.resetDataset(dataset)
    for epoch in range(epoch_start, epoch_end):
        #Shuffle and reset batches
        if shuffle:
            np.random.shuffle(idxlist)
            batchlist = np.split(idxlist, range(batch_size,N,batch_size))
        start_time = time.time()
        bound = 0
        for bnum, batch_idx in enumerate(batchlist):
            batch_idx  = batchlist[bnum]
            batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal = dmm.train_debug(idx=batch_idx)
            if np.any(np.isnan(np.array([batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal]))):
                print 'Warning: you have encountered a NaN'
                import ipdb;ipdb.set_trace()
            bound  += batch_bound
            if epoch%epfreq==0 and bnum%10==0:
                bval = batch_bound/float(dataset['mask'][batch_idx].sum())
                dmm._p(('Bnum: %d, Batch Bound: %.4f, |w|: %.4f, |dw|: %.4f, |w_opt|: %.4f')%(bnum,bval,p_norm, g_norm, opt_norm)) 
                dmm._p(('-veCLL:%.4f, KL:%.4f, anneal:%.4f')%(negCLL, KL, anneal))
        bound /= float(dataset['mask'].sum())
        bound_train_list.append((epoch,bound))
        end_time   = time.time()
        if epoch%epfreq==0:
            dmm._p(('(Ep %d) Bound: %.4f [Took %.4f seconds] ')%(epoch, bound, end_time-start_time))
        if savefreq is not None and epoch%savefreq==0:
            assert savefile is not None, 'expecting savefile'
            dmm._p(('Saving at epoch %d'%epoch))
            dmm._saveModel(fname = savefile+'-EP'+str(epoch))
            intermediate = {}
            if dataset_eval is not None:
                tmpMap = {}
                bound_valid_list.append((epoch, DMM_evaluate.evaluateBound(dmm, dataset_eval, batch_size=batch_size)))
            intermediate['valid_bound'] = np.array(bound_valid_list)
            intermediate['train_bound'] = np.array(bound_train_list)
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
            dmm.resetDataset(dataset)
    retMap = {}
    retMap['train_bound']   = np.array(bound_train_list)
    retMap['valid_bound']   = np.array(bound_valid_list)
    return retMap
