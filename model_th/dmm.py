from __future__ import division
import six.moves.cPickle as pickle
from collections import OrderedDict
import numpy as np
import sys, time, os, gzip, theano,math
sys.path.append('../')
from theano import config
from theano.printing import pydotprint
import theano.tensor as T
from utils.misc import saveHDF5
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils.optimizer import adam,rmsprop
from models.__init__ import BaseModel
from datasets.synthpTheano import updateParamsSynthetic
theano.config.compute_test_value = 'warn'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
""" DEEP MARKOV MODEL"""
class DMM(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        self.scan_updates = []
        super(DMM,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
        assert self.params['nonlinearity']!='maxout','Maxout nonlinearity not supported'
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    """
    Creating parameters for inference and generative model
    """
    def _createParams(self):
        """ Model parameters """
        npWeights = OrderedDict()
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights

    def _createGenerativeParams(self, npWeights):
        """ Create weights/params for generative model """
        DIM_HIDDEN       = self.params['dim_hidden']
        DIM_STOCHASTIC   = self.params['dim_stochastic']
        DIM_HIDDEN_TRANS = DIM_HIDDEN*2
        #Transition Function [MLP]
        for l in range(self.params['transition_layers']):
            dim_input,dim_output = DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS
            if l==0:
                dim_input = self.params['dim_stochastic']
            npWeights['p_trans_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_trans_b_'+str(l)] = self._getWeight((dim_output,))
        if self.params['dim_actions']>0:
            npWeights['p_trans_W_act']     = self._getWeight((self.params['dim_actions'], self.params['dim_stochastic']))
            npWeights['p_trans_b_act']     = self._getWeight((self.params['dim_stochastic'],))

        MU_COV_INP = DIM_HIDDEN_TRANS
        npWeights['p_trans_W_mu']  = self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_b_mu']  = self._getWeight((self.params['dim_stochastic'],))
        npWeights['p_trans_W_cov'] = self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_b_cov'] = self._getWeight((self.params['dim_stochastic'],))
        for l in range(self.params['emission_layers']):
            dim_input,dim_output  = DIM_HIDDEN, DIM_HIDDEN
            if l==0:
                dim_input = self.params['dim_stochastic']
            npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
        if self.params['data_type']=='mixed':
            dim_out = np.where(self.params['feature_types']=='binary')[0].shape[0] + 2*np.where(self.params['feature_types'])[0].shape[0]
        elif self.params['data_type'] == 'real':
            dim_out = self.params['dim_observations']*2
        elif self.params['data_type'] == 'binary':
            dim_out  = self.params['dim_observations']
        else:
            raise ValueError('Bad value for '+str(self.params['data_type']))
        npWeights['p_emis_W_out'] = self._getWeight((self.params['dim_hidden'], dim_out))
        npWeights['p_emis_b_out'] = self._getWeight((dim_out,))

    def _createInferenceParams(self, npWeights):
        """  Create weights/params for inference network """
        #Initial embedding for the inputs
        DIM_INPUT  = self.params['dim_observations']
        RNN_SIZE   = self.params['rnn_size']

        DIM_HIDDEN = RNN_SIZE
        DIM_STOC   = self.params['dim_stochastic']

        #Embed the Input -> RNN_SIZE
        dim_input, dim_output= DIM_INPUT, RNN_SIZE
        npWeights['q_W_input_0'] = self._getWeight((dim_input, dim_output))
        npWeights['q_b_input_0'] = self._getWeight((dim_output,))

        #Setup weights for LSTM
        self._createLSTMWeights(npWeights)

        #Embedding before MF/ST inference model
        if self.params['inference_model']=='mean_field':
            raise ValueError('expecting ST inf')
        elif self.params['inference_model']=='structured':
            DIM_INPUT = self.params['dim_stochastic']
            npWeights['q_W_st_0'] = self._getWeight((DIM_INPUT, self.params['rnn_size']))
            npWeights['q_b_st_0'] = self._getWeight((self.params['rnn_size'],))
            if self.params['dim_actions']>0 and self.params['use_generative_prior'] == 'approx':
                npWeights['q_W_act'] = self._getWeight((self.params['dim_actions'], self.params['rnn_size']))
                npWeights['q_b_act'] = self._getWeight((self.params['rnn_size'],))
        else:
            assert False,'Invalid inference model: '+self.params['inference_model']
        RNN_SIZE = self.params['rnn_size']
        npWeights['q_W_mu']       = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['q_W_cov']      = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_cov']      = self._getWeight((self.params['dim_stochastic'],))

    def _createLSTMWeights(self, npWeights):
        #LSTM L/LR/R w/ orthogonal weight initialization
        suffices_to_build = ['r']
        RNN_SIZE          = self.params['rnn_size']
        assert self.params['rnn_layers']==1,'use 1 layer rnns'
        for suffix in suffices_to_build:
            for l in range(self.params['rnn_layers']):
                npWeights['W_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE*4))
                npWeights['b_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE*4,), scheme='lstm')
                npWeights['U_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE*4),scheme='lstm')

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    """
    Generative Model 
    """
    def _transition(self, z, U=None):
        """
        Transition Function for DMM
        Input:  z [bs x T x dim]
        Output: mu/cov [bs x T x dim]
        """
        if U is not None:
            assert self.params['dim_actions']>0,'U specified but not expected'
            hid = z
            hid += T.dot(U, self.tWeights['p_trans_W_act'])+self.tWeights['p_trans_b_act']
        else:
            hid = z
        for l in range(self.params['transition_layers']):
            hid = self._LinearNL(self.tWeights['p_trans_W_'+str(l)],self.tWeights['p_trans_b_'+str(l)],hid)
        mu     = T.dot(hid, self.tWeights['p_trans_W_mu']) + self.tWeights['p_trans_b_mu']
        cov    = T.nnet.softplus(T.dot(hid, self.tWeights['p_trans_W_cov'])+self.tWeights['p_trans_b_cov'])
        return mu,cov

    def _emission(self, z):
        """
        Emission Function
        Input:  z [bs x T x dim]
        Output: hid [bs x T x dim]
        """
        hid     = z
        for l in range(self.params['emission_layers']):
            hid = self._LinearNL(self.tWeights['p_emis_W_'+str(l)],  self.tWeights['p_emis_b_'+str(l)], hid)
        outp    = T.dot(hid,self.tWeights['p_emis_W_out'])+self.tWeights['p_emis_b_out']
        return outp

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    """
    Negative ELBO [Evidence Lower Bound] 
    """
    def _temporalKL(self, mu_q, cov_q, mu_prior, cov_prior, batchVector = False):
        """
        KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q)
                        + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        """
        assert np.all(cov_q.tag.test_value>0.),'should be positive'
        assert np.all(cov_prior.tag.test_value>0.),'should be positive'
        diff_mu = mu_prior-mu_q
        KL_t    = T.log(cov_prior)-T.log(cov_q) - 1. + cov_q/cov_prior + diff_mu**2/cov_prior
        KLvec   = (0.5*KL_t.sum(2)).sum(1,keepdims=True)
        if batchVector:
            return KLvec
        else:
            return KLvec.sum()

    def _neg_elbo(self, X, B, M, U = None, anneal = 1., dropout_prob = 0., additional = None):
        z_q, mu_q, cov_q   = self._q_z_x(X, U= U, dropout_prob = dropout_prob)
        mu_trans, cov_trans= self._transition(z_q, U = U)
        """
        Obtain initial prior distribution
        """
        B_mu               = T.dot(B, self.tWeights['B_mu_W'])[:,None,:] 
        B_cov              = T.nnet.softplus(T.dot(B, self.tWeights['B_cov_W']))[:,None,:]
        mu_prior           = T.concatenate([B_mu, mu_trans[:,:-1,:]],axis=1)
        cov_prior          = T.concatenate([B_cov,cov_trans[:,:-1,:]],axis=1)
        KL                 = self._temporalKL(mu_q, cov_q, mu_prior, cov_prior)
        hid_out            = self._emission(z_q)
	params             = {}
        if self.params['data_type'] == 'mixed':
            nll            = self._nll_mixed(hid_out, X, mask=M, ftypes = self.params['feature_types'], params = params).sum()
        elif self.params['data_type'] == 'binary':
            nll            = self._nll_binary(hid_out, X, mask = M, params= params).sum()
        elif self.params['data_type']=='real':
            dim_obs        = self.params['dim_observations']
            nll            = self._nll_gaussian(hid_out[:,:,:dim_obs], hid_out[:,:,dim_obs:dim_obs*2], X, mask = M, params=params).sum()
        else:
            raise ValueError('Invalid Data Type'+str(self.params['data_type']))
        #Evaluate negative ELBO
        neg_elbo   = nll+anneal*KL
        if additional is not None:
            additional['nll']    = nll
            additional['kl']     = KL
            additional['b_mu']   = B_mu
            additional['b_cov']  = B_cov
            additional['mu_q']   = mu_q
            additional['cov_q']  = cov_q
            additional['z_q']    = z_q
            additional['mu_t']   = mu_trans
            additional['cov_t']  = cov_trans
	    for k in params:
		additional[k] = params[k] 
        return neg_elbo
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

    """
    Recognition Model 
    """
    def _aggregateLSTM(self, hidden_state):
        """
        LSTM hidden layer [T x bs x dim] 
        z [bs x T x dim], mu [bs x T x dim], cov [bs x T x dim]
        """
        if self.params['dim_stochastic']==1:
            raise ValueError('dim_stochastic must be larger than 1 dimension')
        def structuredApproximation(h_t, eps_t, z_prev,
                                    q_W_st_0, q_b_st_0,
                                    q_W_mu, q_b_mu,
                                    q_W_cov,q_b_cov):
            h_next     = T.tanh(T.dot(z_prev,q_W_st_0)+q_b_st_0)
            h_next     = (1./2.)*(h_t+h_next)
            mu_t       = T.dot(h_next,q_W_mu)+q_b_mu
            cov_t      = T.nnet.softplus(T.dot(h_next,q_W_cov)+q_b_cov)
            z_t        = mu_t+T.sqrt(cov_t)*eps_t
            return z_t, mu_t, cov_t
        # eps: [T x bs x dim_stochastic]
        eps         = self.srng.normal(size=(hidden_state.shape[0],hidden_state.shape[1],self.params['dim_stochastic']))
        z0          = T.zeros((eps.shape[1], self.params['dim_stochastic']))
        rval, _     = theano.scan(structuredApproximation, sequences=[hidden_state, eps],
                                    outputs_info=[z0, None,None],
                                    non_sequences=[self.tWeights[k] for k in ['q_W_st_0', 'q_b_st_0']]+
                                                  [self.tWeights[k] for k in ['q_W_mu','q_b_mu','q_W_cov','q_b_cov']],
                                    name='structuredApproximation')
        z, mu, cov  = rval[0].swapaxes(0,1), rval[1].swapaxes(0,1), rval[2].swapaxes(0,1)
        return z, mu, cov

    def _pog(self, mu_1,cov_1, mu_2,cov_2):
        cov_f = T.sqrt((cov_1*cov_2)/(cov_1+cov_2))
        mu_f  = (mu_1*cov_2+mu_2*cov_1)/(cov_1+cov_2) 
        return mu_f, cov_f
    def _aggregateLSTM_U(self, hidden_state, U=None):
        assert U is not None,'expecting U'
        if self.params['dim_stochastic']==1:
            raise ValueError('dim_stochastic must be larger than 1 dimension')
        """
        Use the true prior
            * Gradients for predicting z_2 will be propagated 
        """
        def st_true(h_t, eps_t, u_t, z_prev, *params):
            q_W_mu, q_b_mu, q_W_cov, q_b_cov = params[:4]
            p_trans_W_act, p_trans_b_act     = params[4:6]
            p_trans_W_mu, p_trans_b_mu       = params[6:8]
            p_trans_W_cov, p_trans_b_cov     = params[8:10]
            ctr     = 10 
            tparams = {}
            for l in range(self.params['transition_layers']):
                tparams['p_trans_W_'+str(l)] = params[ctr]
                ctr += 1
                tparams['p_trans_b_'+str(l)] = params[ctr] 
                ctr += 1
            assert ctr == len(params),'Bad length found'
            """
            Transition fxn
            """
            hid  = z_prev
            hid += T.dot(u_t, p_trans_W_act) + p_trans_b_act
            for l in range(self.params['transition_layers']):
                hid      = self._LinearNL(tparams['p_trans_W_'+str(l)], tparams['p_trans_b_'+str(l)], hid)
            mu_trans     = T.dot(hid, p_trans_W_mu) + p_trans_b_mu
            cov_trans    = T.nnet.softplus(T.dot(hid, p_trans_W_cov)+p_trans_b_cov)
            """
            RNN hidden state
            """
            mu_t       = T.dot(h_t,q_W_mu)+q_b_mu
            cov_t      = T.nnet.softplus(T.dot(h_t,q_W_cov)+q_b_cov)
            #POG Approximation
            mu_f, cov_f= self._pog(mu_trans, cov_trans, mu_t, cov_t) 
            z_t        = mu_f+T.sqrt(cov_f)*eps_t
            return z_t, mu_f, cov_f
        """
        Use an approximation to the prior
            * Use the targets but not necessarily the true prior distribution
            * This has the effect of making the value of z_2 depend on the target
        """
        def st_approx(h_t, eps_t, u_t, z_prev, *params):
            q_W_st, q_b_st  = params[0:2] 
            q_W_mu, q_b_mu  = params[2:4]
            q_W_cov, q_b_cov= params[4:6]
            q_W_act, q_b_act= params[6:8]
            h_z        = T.tanh(T.dot(z_prev,q_W_st)+q_b_st)
            h_act      = T.tanh(T.dot(u_t,q_W_act)+q_b_act)
            h_next     = (1./3.)*(h_t+h_z+h_act)
            mu_t       = T.dot(h_next,q_W_mu)+q_b_mu
            cov_t      = T.nnet.softplus(T.dot(h_next,q_W_cov)+q_b_cov)
            z_t        = mu_t+T.sqrt(cov_t)*eps_t
            return z_t, mu_t, cov_t
        # eps: [T x bs x dim_stochastic]
        eps         = self.srng.normal(size=(hidden_state.shape[0],hidden_state.shape[1],self.params['dim_stochastic']))
        z0          = T.zeros((eps.shape[1], self.params['dim_stochastic']))
        if self.params['use_generative_prior']=='true':
            non_seq = [self.tWeights[k] for k in ['q_W_mu','q_b_mu','q_W_cov','q_b_cov']]
            non_seq+= [self.tWeights[k] for k in ['p_trans_W_act','p_trans_b_act','p_trans_W_mu','p_trans_b_mu','p_trans_W_cov','p_trans_b_cov']]
            for l in range(self.params['transition_layers']):
                non_seq+= [self.tWeights['p_trans_W_'+str(l)],self.tWeights['p_trans_b_'+str(l)]]
            scan_fxn=st_true
        elif self.params['use_generative_prior']=='approx':
            non_seq = [self.tWeights[k] for k in ['q_W_st_0', 'q_b_st_0', 'q_W_mu','q_b_mu','q_W_cov','q_b_cov','q_W_act','q_b_act']]
            scan_fxn=st_approx 
        else:
            raise NotImplemented('Should not reach here')
        rval, _     = theano.scan(scan_fxn, sequences=[hidden_state, eps, U.swapaxes(0,1)],
                                    outputs_info=[z0, None,None],
                                    non_sequences=non_seq,
                                    name='structuredApproximation')
        z, mu, cov  = rval[0].swapaxes(0,1), rval[1].swapaxes(0,1), rval[2].swapaxes(0,1)
        return z, mu, cov
    def _q_z_x(self, X, U = None, dropout_prob = 0.):
        """
        Inference
        X: nbatch x time x dim_observations 
        Returns: z_q (nbatch x time x dim_stochastic), mu_q (nbatch x time x dim_stochastic) and cov_q (nbatch x time x dim_stochastic)
        """
        self._p('Building with RNN dropout:'+str(dropout_prob))
        embedding         = self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], X)
        hidden_state      = self._LSTMlayer(embedding, 'r', dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])
        if self.params['dim_actions']>0 and self.params['use_generative_prior'] in ['true','approx']:
            assert U is not None,'expecting U' 
            assert self.params['dim_actions']>0,'non 0 actions'
            z_q, mu_q, cov_q  = self._aggregateLSTM_U(hidden_state, U = U)
        else:
            z_q, mu_q, cov_q  = self._aggregateLSTM(hidden_state)
        return z_q, mu_q, cov_q
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

    """
    Resetting Datasets
    """
    def resetDataset(self, dataset, quiet=False):
        if self.params['dim_actions'] > 0:
            if not quiet:
                ddim, mdim, bdim, udim= self.dimData()
                self._p('Original dim:'+str(ddim)+', '+str(mdim)+','+str(bdim)+','+str(udim))#+','+str(u_mdim))
            newX, newM = dataset['features']['tensor'].astype(config.floatX), dataset['features']['obs_tensor'].astype(config.floatX)
            newB       = dataset['baselines']['tensor'].astype(config.floatX)
            newU       = dataset['treatments']['tensor'].toarray().astype(config.floatX)
            #newU_mask  = dataset['treatments']['obs_tensor'].astype(config.floatX)
            self.setData(newX=newX, newMask = newM, newB = newB, newU = newU)#, newU_mask=newU_mask)
            if not quiet:
                ddim, mdim, bdim, udim = self.dimData()
                self._p('New dim:'+str(ddim)+', '+str(mdim)+','+str(bdim)+','+str(udim))#+','+str(u_mdim))
        else:
            if not quiet:
                ddim, mdim, bdim = self.dimData()
                self._p('Original dim:'+str(ddim)+', '+str(mdim)+', '+str(bdim))
            newX, newM = dataset['features']['tensor'].astype(config.floatX), dataset['features']['obs_tensor'].astype(config.floatX)
            newB       = dataset['baselines']['tensor'].astype(config.floatX)
            self.setData(newX=newX, newMask = newM, newB = newB)
            if not quiet:
                ddim, mdim, bdim = self.dimData()
                self._p('New dim:'+str(ddim)+', '+str(mdim)+', '+str(bdim))
    """ Building Model """
    def _buildModel(self):
        """ High level function to build and setup theano functions """
        idx                = T.vector('idx',dtype='int64')
        idx.tag.test_value = np.array([0,1]).astype('int64')
        X_init             = np.random.uniform(0,1,size=(3,5,self.params['dim_observations'])).astype(config.floatX)
        M_init             = ((X_init>0.5)*1.).astype(config.floatX) 
        B_init             = np.random.uniform(0,1,size=(3,self.params['dim_baselines'])).astype(config.floatX)
        self.dataset       = theano.shared(X_init)
        self.dataset_b     = theano.shared(B_init)
        self.mask          = theano.shared(M_init)
        U_o, U_m, self.dataset_u, self.dataset_u_mask= None, None, None, None
        if self.params['dim_actions']>0:
            U_init         = np.random.uniform(0,1,size=(3,5,self.params['dim_actions'])).astype(config.floatX)
            self.dataset_u       = theano.shared(U_init)
            #self.dataset_u_mask  = theano.shared(((U_init>0.5)*1.).astype('float32'))
            U_o            = self.dataset_u[idx]
            #U_m            = self.dataset_u_mask[idx]
        X_o                = self.dataset[idx]
        B_o                = self.dataset_b[idx]
        M_o                = self.mask[idx]
        #Support for variable length sequences - ignore for now
        #maxidx            = T.cast(M_o.sum(1).max(),'int64')
        X                  = X_o#[:,:maxidx,:]
        B                  = B_o#[:,:maxidx,:]
        M                  = M_o#[:,:maxidx]
        U                  = None
        #U_mask             = None
        if self.params['dim_actions']>0:
            U              = U_o#[:,:maxidx,:]
            #U_mask         = U_m#[:,:maxidx,:]
        newX, newMask, newB= T.tensor3('newX',dtype=config.floatX), T.tensor3('newMask',dtype=config.floatX), T.matrix('newB',dtype=config.floatX)
        if self.params['dim_actions']>0:
            newU               = T.tensor3('newU',dtype=config.floatX)
            #newU_mask          = T.tensor3('newU_mask',dtype=config.floatX)
            self.setData       = theano.function([newX, newMask, newU,newB],None,
                    updates=[(self.dataset,newX),(self.mask,newMask), (self.dataset_u,newU), (self.dataset_b, newB)])
            self.dimData       = theano.function([],[self.dataset.shape, self.mask.shape, self.dataset_b.shape, self.dataset_u.shape])
        else:
            self.setData       = theano.function([newX, newMask, newB],None,updates=[(self.dataset,newX),(self.mask,newMask), (self.dataset_b, newB)])
            self.dimData       = theano.function([],[self.dataset.shape, self.mask.shape, self.dataset_b.shape])
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('anneal', np.asarray(0.01,dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)
        lr             = self.tWeights['lr']
        anneal         = self.tWeights['anneal']
        iteration_t    = self.tWeights['update_ctr']
        anneal_div     = 1000.
        if 'anneal_rate' in self.params:
            self._p('Anneal = 1 in '+str(self.params['anneal_rate'])+' param. updates')
            anneal_div = self.params['anneal_rate']
        anneal_update  = [(iteration_t, iteration_t+1), (anneal,T.switch(0.01+iteration_t/anneal_div>1,1,0.01+iteration_t/anneal_div))]
        fxn_inputs = [idx]
        if not self.params['validate_only']:
            traindict  = {}
            train_cost = self._neg_elbo(X, B, M, U = U, anneal = anneal, dropout_prob = self.params['rnn_dropout'], additional=traindict)
            #Get updates from optimizer
            model_params             = self._getModelParams()
            optimizer_up, norm_list  = self._setupOptimizer(train_cost, model_params,lr = lr,
                                                            reg_type =self.params['reg_type'],
                                                            reg_spec =self.params['reg_spec'],
                                                            reg_value= self.params['reg_value'],
                                                            divide_grad = None, 
                                                            grad_norm = 1.)
            #Add annealing updates
            optimizer_up += anneal_update+self.updates
            self._p(str(len(self.updates))+' other updates')
            ############# Setup train & evaluate functions ###########
            self.train_debug = theano.function(fxn_inputs,[train_cost,norm_list[0],norm_list[1],
                norm_list[2], traindict['nll'], traindict['kl'], anneal.sum()], updates = optimizer_up, name='Train (with Debug)')
        #Updates ack
        self.updates_ack         = True
	evaldict                 = {}
        eval_cost                = self._neg_elbo(X, B, M, U = U, anneal = anneal, dropout_prob = 0., additional= evaldict)
        self.evaluate            = theano.function([idx], eval_cost, name = 'Evaluate Bound',allow_input_downcast=True)
        self.posterior_inference = theano.function([idx], [evaldict['z_q'], evaldict['mu_q'], evaldict['cov_q']], name='Posterior Inference',allow_input_downcast=True)
        self.emission_fxn        = theano.function([evaldict['z_q']], [evaldict['bin_prob'], evaldict['real_mu'], evaldict['real_logcov']] , name='Emission',allow_input_downcast=True)
	self.init_prior          = theano.function([B],[evaldict['b_mu'],evaldict['b_cov']], name = 'Initial Prior',allow_input_downcast=True)
	if self.params['dim_actions']>0:
        	self.transition_fxn = theano.function([evaldict['z_q'], U], [evaldict['mu_t'], evaldict['cov_t']], name='Transition Function', allow_input_downcast=True)
	else:
        	self.transition_fxn = theano.function([evaldict['z_q']], [evaldict['mu_t'], evaldict['cov_t']], name='Transition Function',allow_input_downcast=True)
        self._p('Completed DMM setup')
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
if __name__=='__main__':
    """ use this to check compilation for various options"""
    from parse_args import params
    params['data_type'] = 'binary'
    params['dim_observations']  = 10
    dmm = DMM(params, paramFile = 'tmp')
    os.unlink('tmp')
    import ipdb;ipdb.set_trace()
