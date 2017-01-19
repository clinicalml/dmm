""" Functions for learning a DMM object """
import evaluate as DMM_evaluate
import numpy as np
from utils.misc import saveHDF5
import time
from theano import config

def learn(dmm, dataset, mask, epoch_start=0, epoch_end=1000, 
          batch_size=200, shuffle=True, savefreq=None, savefile = None, 
          dataset_eval = None, mask_eval = None, replicate_K =None,
          normalization = 'frame'):
    """
                                            Train DMM
    """
    assert not dmm.params['validate_only'],'cannot learn in validate only mode'
    assert len(dataset.shape)==3,'Expecting 3D tensor for data'
    assert dataset.shape[2]==dmm.params['dim_observations'],'Dim observations not valid'
    N = dataset.shape[0]
    idxlist   = range(N)
    batchlist = np.split(idxlist, range(batch_size,N,batch_size))

    bound_train_list,bound_valid_list,bound_tsbn_list,nll_valid_list = [],[],[],[]
    p_norm, g_norm, opt_norm = None, None, None

    #Lists used to track quantities for synthetic experiments
    mu_list_train, cov_list_train, mu_list_valid, cov_list_valid = [],[],[],[]
    model_params = {} 
    epfreq = 1
    #Set data
    dmm.resetDataset(dataset, mask)
    for epoch in range(epoch_start, epoch_end):
        #Shuffle
        if shuffle:
            np.random.shuffle(idxlist)
            batchlist = np.split(idxlist, range(batch_size,N,batch_size))
        #Always shuffle order the batches are presented in
        np.random.shuffle(batchlist)

        start_time = time.time()
        bound = 0
        for bnum, batch_idx in enumerate(batchlist):
            batch_idx  = batchlist[bnum]
            batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal = dmm.train_debug(idx=batch_idx)

            #Number of frames
            M_sum = mask[batch_idx].sum()
            #Correction for replicating batch
            if replicate_K is not None:
                batch_bound, negCLL, KL = batch_bound/replicate_K, negCLL/replicate_K, KL/replicate_K, 
                M_sum   = M_sum/replicate_K
            #Update bound
            bound  += batch_bound
            ### Display ###
            if epoch%epfreq==0 and bnum%10==0:
                if normalization=='frame':
                    bval = batch_bound/float(M_sum)
                elif normalization=='sequence':
                    bval = batch_bound/float(X.shape[0])
                else:
                    assert False,'Invalid normalization'
                dmm._p(('Bnum: %d, Batch Bound: %.4f, |w|: %.4f, |dw|: %.4f, |w_opt|: %.4f')%(bnum,bval,p_norm, g_norm, opt_norm)) 
                dmm._p(('-veCLL:%.4f, KL:%.4f, anneal:%.4f')%(negCLL, KL, anneal))
        if normalization=='frame':
            val =mask.sum()
            if replicate_K is not None:
                val/=float(replicate_K)
            bound /= float(val)
        elif normalization=='sequence':
            bound /= float(N)
        else:
            assert False,'Invalid normalization'
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
            if dataset_eval is not None and mask_eval is not None:
                tmpMap = {}
                bound_valid_list.append(
                    (epoch, 
                     DMM_evaluate.evaluateBound(dmm, dataset_eval, mask_eval, batch_size=batch_size, 
                                              additional = tmpMap, normalization=normalization)))
                bound_tsbn_list.append((epoch, tmpMap['tsbn_bound']))
                nll_valid_list.append(
                    DMM_evaluate.impSamplingNLL(dmm, dataset_eval, mask_eval, batch_size,
                                                                  normalization=normalization))
            intermediate['valid_bound'] = np.array(bound_valid_list)
            intermediate['train_bound'] = np.array(bound_train_list)
            intermediate['tsbn_bound']  = np.array(bound_tsbn_list)
            intermediate['valid_nll']  = np.array(nll_valid_list)
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
            ### Update X in the computational flow_graph to point to training data
            dmm.resetDataset(dataset, mask)
    #Final information to be collected
    retMap = {}
    retMap['train_bound']   = np.array(bound_train_list)
    retMap['valid_bound']   = np.array(bound_valid_list)
    retMap['tsbn_bound']   = np.array(bound_tsbn_list)
    retMap['valid_nll']  = np.array(nll_valid_list)
    return retMap
