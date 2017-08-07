""" Functions for learning a DMM object """
import evaluate as DMM_evaluate
import numpy as np
from utils.misc import saveHDF5
import time
from theano import config

def learn(dmm, dataset, epoch_start=0, epoch_end=1000, 
          batch_size=200, shuffle=True, savefreq=None, savefile = None, 
          dataset_eval = None):
    """ Train DMM """
    assert not dmm.params['validate_only'],'cannot learn in validate only mode'
    N         = dataset['features']['tensor'].shape[0]
    idxlist   = range(N)
    batchlist = np.split(idxlist, range(batch_size,N,batch_size))
    bound_train_list,bound_valid_list = [],[]
    epfreq = 1
    #Set data
    dmm.resetDataset(dataset)
    for epoch in range(epoch_start, epoch_end):
        #Shuffle
        if shuffle:
            np.random.shuffle(idxlist)
            batchlist = np.split(idxlist, range(batch_size,N,batch_size))
        start_time = time.time()
        bound = 0
        for bnum, batch_idx in enumerate(batchlist):
            batch_idx  = batchlist[bnum]
            #print bnum,'\n'.join(['%.4f'%np.linalg.norm((dmm.tWeights[k]*1.).eval())+' '+k for k in dmm.tWeights])
            #if np.any(np.isnan(np.array([np.linalg.norm((dmm.tWeights[k]*1.).eval()) for k in dmm.tWeights]))):
            #    import ipdb;ipdb.set_trace()
            #if bnum==5:
            #    import ipdb;ipdb.set_trace()
            batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal = dmm.train_debug(idx=batch_idx)
            if np.any(np.isnan(np.array([batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal]))):
                import ipdb;ipdb.set_trace()
            #if np.any(np.isnan(np.array([np.linalg.norm((dmm.tWeights[k]*1.).eval()) for k in dmm.tWeights]))):
            #    import ipdb;ipdb.set_trace()
            bound  += batch_bound
            ### Display ###
            if epoch%epfreq==0 and bnum%10==0:
                bval = batch_bound/float(dataset['features']['obs_tensor'][batch_idx].sum())
                dmm._p(('Bnum: %d, Batch Bound: %.4f, |w|: %.4f, |dw|: %.4f, |w_opt|: %.4f')%(bnum,bval,p_norm, g_norm, opt_norm)) 
                dmm._p(('-veCLL:%.4f, KL:%.4f, anneal:%.4f')%(negCLL, KL, anneal))
        bound /= float(dataset['features']['obs_tensor'].sum())
        bound_train_list.append((epoch,bound))
        end_time   = time.time()
        if epoch%epfreq==0:
            dmm._p(('(Ep %d) Bound: %.4f [Took %.4f seconds] ')%(epoch, bound, end_time-start_time))
        #Save at intermediate stages
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
    #Final information to be collected
    retMap = {}
    retMap['train_bound']   = np.array(bound_train_list)
    retMap['valid_bound']   = np.array(bound_valid_list)
    return retMap
