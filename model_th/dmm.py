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
theano.config.compute_test_value = 'warn'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
""" DEEP MARKOV MODEL [previously: DEEP KALMAN FILTER] """
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
        np.random.seed(self.params['seed'])
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights

    def _createGenerativeParams(self, npWeights):
        """ Create weights/params for generative model """
        DIM_HIDDEN     = self.params['dim_hidden']
        DIM_STOCHASTIC = self.params['dim_stochastic']
        """
        Transition Function _ MLP 
        """
        for l in range(self.params['transition_layers']):
            dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
            if l==0:
                dim_input     = self.params['dim_stochastic']
            npWeights['p_trans_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_trans_b_'+str(l)] = self._getWeight((dim_output,))
            if self.params['transition_type']=='gated':
                npWeights['p_trans_gate_W_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['p_trans_gate_b_'+str(l)] = self._getWeight((dim_output,))
        MU_COV_INP = self.params['dim_hidden']
        if self.params['transition_layers']==0:
            MU_COV_INP     = self.params['dim_stochastic']
        if self.params['transition_type']=='gated':
            npWeights['p_trans_z_W']     = self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
            npWeights['p_trans_gate_W']  = self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_W_mu']  = self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_b_mu']  = self._getWeight((self.params['dim_stochastic'],))
        npWeights['p_trans_W_cov'] = self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_b_cov'] = self._getWeight((self.params['dim_stochastic'],))

        """ 
        Emission Function 
        """
        for l in range(self.params['emission_layers']):
            dim_input,dim_output  = DIM_HIDDEN, DIM_HIDDEN
            if l==0:
                dim_input = self.params['dim_stochastic']
            npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
        if self.params['data_type'] == 'real':
            dim_out = self.params['dim_observations']*2
        elif self.params['data_type'] == 'binary':
            dim_out  = self.params['dim_observations']
        else:
            raise ValueError('Bad value for '+str(self.params['data_type']))
        dim_in = self.params['dim_hidden']
        if self.params['emission_layers']==0:
            dim_in = self.params['dim_stochastic']
        npWeights['p_emis_W_out'] = self._getWeight((dim_in, dim_out))
        npWeights['p_emis_b_out'] = self._getWeight((dim_out,))

    def _createInferenceParams(self, npWeights):
        """  Create weights/params for inference network """
        #Initial embedding for the inputs
        DIM_INPUT  = self.params['dim_observations']
        RNN_SIZE   = self.params['rnn_size']
        DIM_HIDDEN = RNN_SIZE
        DIM_STOC   = self.params['dim_stochastic']

        #Step 1: Params for initial embedding of input 
        dim_input, dim_output= DIM_INPUT, RNN_SIZE
        npWeights['q_W_input_0'] = self._getWeight((dim_input, dim_output))
        npWeights['q_b_input_0'] = self._getWeight((dim_output,))

        #Step 2: RNN/LSTM params
        self._createLSTMWeights(npWeights)
        
        #Step 3: Parameters for combiner function
        assert self.params['inference_model'] in ['LR','R'],'Invalid inference model'
        if self.params['use_generative_prior']=='approx':
            DIM_INPUT = self.params['dim_stochastic']
            npWeights['q_W_st'] = self._getWeight((DIM_INPUT, self.params['rnn_size']))
            npWeights['q_b_st'] = self._getWeight((self.params['rnn_size'],))
        RNN_SIZE = self.params['rnn_size']
        npWeights['q_W_mu']       = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['q_W_cov']      = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_cov']      = self._getWeight((self.params['dim_stochastic'],))

    def _createLSTMWeights(self, npWeights):
        suffices_to_build = ['r']
        if self.params['inference_model']=='LR':
            suffices_to_build.append('l')
        RNN_SIZE          = self.params['rnn_size']
        for suffix in suffices_to_build:
            if self.params['rnn_cell']=='lstm':
                npWeights['W_lstm_'+suffix] = self._getWeight((RNN_SIZE,RNN_SIZE*4))
                npWeights['b_lstm_'+suffix] = self._getWeight((RNN_SIZE*4,), scheme='lstm')
                npWeights['U_lstm_'+suffix] = self._getWeight((RNN_SIZE,RNN_SIZE*4),scheme='lstm')
            elif self.params['rnn_cell']=='rnn':
                npWeights['W_rnn_'+suffix] = self._getWeight((RNN_SIZE,RNN_SIZE))
                npWeights['b_rnn_'+suffix] = self._getWeight((RNN_SIZE,), scheme='lstm')
                npWeights['U_rnn_'+suffix] = self._getWeight((RNN_SIZE,RNN_SIZE),scheme='orthogonal')
            else:
                raise ValueError('Invalid setting for RNN cell')

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _transition(self, z, trans_params = None):
        """ Transition Function for DMM
        Input:  z [bs x T x dim]
        Output: mu/cov [bs x T x dim] """
        if trans_params is None:
            trans_params = self.tWeights
        hid    = z
        hid_g  = z
        for l in range(self.params['transition_layers']):
            hid= self._LinearNL(trans_params['p_trans_W_'+str(l)],trans_params['p_trans_b_'+str(l)],hid)
            if self.params['transition_type']=='gated': 
                hid_g = self._LinearNL(trans_params['p_trans_gate_W_'+str(l)],trans_params['p_trans_gate_b_'+str(l)],hid_g)
        mu_prop= T.dot(hid, trans_params['p_trans_W_mu']) + trans_params['p_trans_b_mu']
        if self.params['transition_type']=='gated':
            gate   = T.nnet.sigmoid(T.dot(hid_g, trans_params['p_trans_gate_W']))
            mu     = gate*mu_prop + (1-gate)*T.dot(z, trans_params['p_trans_z_W']) 
        else:
            mu     = mu_prop
        cov    = T.nnet.softplus(T.dot(hid, trans_params['p_trans_W_cov'])+trans_params['p_trans_b_cov'])
        return mu, cov

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
    def _temporalKL(self, mu_q, cov_q, mu_prior, cov_prior, mask):
        """
        KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q)
                        + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        """
        assert np.all(cov_q.tag.test_value>0.),'should be positive'
        assert np.all(cov_prior.tag.test_value>0.),'should be positive'
        diff_mu = mu_prior-mu_q
        KL      = T.log(cov_prior)-T.log(cov_q) - 1. + cov_q/cov_prior + diff_mu**2/cov_prior
        KL_t    = 0.5*KL.sum(2)
        KLmasked  = (KL_t*mask)
        return KLmasked.sum()

    def _neg_elbo(self, X, M, anneal = 1., dropout_prob = 0., additional = None):
        z_q, mu_q, cov_q   = self._q_z_x(X, mask = M, dropout_prob = dropout_prob, anneal = anneal)
        mu_trans, cov_trans= self._transition(z_q)
        mu_prior           = T.concatenate([T.zeros_like(mu_trans[:,[0],:]), mu_trans[:,:-1,:]], axis=1)
        cov_prior          = T.concatenate([T.ones_like(mu_trans[:,[0],:]),cov_trans[:,:-1,:]],axis=1)
        KL                 = self._temporalKL(mu_q, cov_q, mu_prior, cov_prior, mask = M)
        hid_out            = self._emission(z_q)
	params             = {}
        if self.params['data_type'] == 'binary':
            nll_mat        = self._nll_binary(hid_out, X, mask = M, params= params)
        elif self.params['data_type']=='real':
            dim_obs        = self.params['dim_observations']
            mu_hid         = hid_out[:,:,:dim_obs]
            logcov_hid     = hid_out[:,:,dim_obs:dim_obs*2]
            nll_mat        = self._nll_gaussian(mu_hid, logcov_hid, X, mask = M, params=params)
        else:
            raise ValueError('Invalid Data Type'+str(self.params['data_type']))
        nll            = nll_mat.sum()
        #Evaluate negative ELBO
        neg_elbo       = nll+anneal*KL
        if additional is not None:
            additional['hid_out']= hid_out
            additional['nll_mat']= nll_mat
            additional['nll_batch']= nll_mat.sum((1,2))
            additional['nll_feat']= nll_mat.sum((0,2))
            additional['nll']    = nll
            additional['kl']     = KL
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
    def _pog(self, mu_1,cov_1, mu_2,cov_2):
        cov_f = T.sqrt((cov_1*cov_2)/(cov_1+cov_2))
        mu_f  = (mu_1*cov_2+mu_2*cov_1)/(cov_1+cov_2) 
        return mu_f, cov_f

    def _aggregateLSTM(self, hidden_state):
        """
        LSTM hidden layer [T x bs x dim] 
        z [bs x T x dim], mu [bs x T x dim], cov [bs x T x dim]
        """
        if self.params['dim_stochastic']==1:
            raise ValueError('dim_stochastic must be larger than 1 dimension')
        def st_approx(h_t, eps_t, z_prev, *params):
            tParams    = OrderedDict()
            for p in params:
                tParams[p.name] = p
            h_next     = T.tanh(T.dot(z_prev,tParams['q_W_st'])+tParams['q_b_st'])
            h_next     = (1./2.)*(h_t+h_next)
            mu_t       = T.dot(h_next,tParams['q_W_mu'])+tParams['q_b_mu']
            cov_t      = T.nnet.softplus(T.dot(h_next, tParams['q_W_cov'])+tParams['q_b_cov'])
            z_t        = mu_t+T.sqrt(cov_t)*eps_t
            return z_t, mu_t, cov_t
        def st_true(h_t, eps_t, z_prev, *params):
            tParams    = OrderedDict()
            for p in params:
                tParams[p.name] = p
            mu_trans, cov_trans = self._transition(z_prev, trans_params = tParams)
            mu_t       = T.dot(h_t,tParams['q_W_mu'])+tParams['q_b_mu']
            cov_t      = T.nnet.softplus(T.dot(h_t, tParams['q_W_cov'])+tParams['q_b_cov'])
            mu_f, cov_f= self._pog(mu_trans, cov_trans, mu_t, cov_t)
            z_f        = mu_f+T.sqrt(cov_f)*eps_t
            return z_f, mu_f, cov_f
        # eps: [T x bs x dim_stochastic]
        eps         = self.srng.normal(size=(hidden_state.shape[0],hidden_state.shape[1],self.params['dim_stochastic']))
        z0          = self.srng.normal(size=(eps.shape[1], eps.shape[-1]))
        if self.params['use_generative_prior'] == 'true':
            non_seq = [self.tWeights[k] for k in ['q_W_mu','q_b_mu','q_W_cov','q_b_cov']]+[self.tWeights[k] for k in self.tWeights if '_trans_' in k]
            step_fxn= st_true
        else:
            non_seq = [self.tWeights[k] for k in ['q_W_st', 'q_b_st','q_W_mu','q_b_mu','q_W_cov','q_b_cov']]
            step_fxn= st_approx
        rval, _     = theano.scan(step_fxn, sequences=[hidden_state, eps],
                                    outputs_info=[z0, None,None],
                                    non_sequences=non_seq, 
                                    name='structuredApproximation')
        z, mu, cov  = rval[0].swapaxes(0,1), rval[1].swapaxes(0,1), rval[2].swapaxes(0,1)
        return z, mu, cov

    def _LSTM_RNN_layer(self, inp, suffix, temporalMask = None, dropout_prob=0., RNN_SIZE = None):
        self._p(('In _LSTM_RNN_layer with dropout %.4f')%(dropout_prob))
        assert suffix=='r' or suffix=='l','Invalid suffix: '+suffix
        def _slice(mat, n, dim):
            if mat.ndim == 3:
                return mat[:, :, n * dim:(n + 1) * dim]
            return mat[:, n * dim:(n + 1) * dim]
        def _lstm_layer(x_, t_m_, h_, c_, lstm_U):
            preact = T.dot(h_, lstm_U)
            preact += x_
            i = T.nnet.sigmoid(_slice(preact, 0, RNN_SIZE))
            f = T.nnet.sigmoid(_slice(preact, 1, RNN_SIZE))
            o = T.nnet.sigmoid(_slice(preact, 2, RNN_SIZE))
            c = T.tanh(_slice(preact, 3, RNN_SIZE))
            # c and h are only updated if the current time 
            # step contains atleast an observed feature 
            obs_t = t_m_[:,None]
            c_new = f * c_ + i * c
            c = c_new*obs_t+ (1-obs_t)*c_
            h_new = o * T.tanh(c)
            h = h_new*obs_t+ (1-obs_t)*h_
            return h, c  

        def _rnn_layer(x_, t_m_, h_, lstm_U):
            h_next  = T.dot(h_, lstm_U) + x_
            obs_t = t_m_[:,None]
            h_out   = obs_t*(T.tanh(h_next)) + (1-obs_t)*h_
            return h_out 
        rnn_cell       = self.params['rnn_cell']
        if rnn_cell=='lstm':
            stepfxn    = _lstm_layer
        elif rnn_cell == 'rnn':
            stepfxn    = _rnn_layer
        else:
            raise ValueError('Invalid cell: '+rnn_cell)

        lstm_embed = T.dot(inp.swapaxes(0,1),self.tWeights['W_'+rnn_cell+'_'+suffix])+ self.tWeights['b_'+rnn_cell+'_'+suffix]
        nsteps     = lstm_embed.shape[0]
        n_samples  = lstm_embed.shape[1]

        if self.params['rnn_cell']=='lstm':
            o_info = [T.zeros((n_samples,RNN_SIZE)), T.ones((n_samples,RNN_SIZE))]
        else:
            o_info = [T.zeros((n_samples,RNN_SIZE))]
        n_seq      = [self.tWeights['U_'+rnn_cell+'_'+suffix]]
        lstm_input = lstm_embed
        if temporalMask is None: 
            tMask      = T.ones((nsteps, n_samples))
        else:
            tMask      = temporalMask.swapaxes(0,1)
        if suffix=='r':
            lstm_input = lstm_input[::-1]
            tMask      = tMask[::-1]
        rval, _= theano.scan(stepfxn,
                              sequences=[lstm_input, tMask],
                              outputs_info  = o_info,
                              non_sequences = n_seq,
                              name='RNN_LSTM_'+suffix,
                              n_steps=nsteps)
        if self.params['rnn_cell']=='lstm': 
            lstm_output =  rval[0]
        else:
            lstm_output =  rval
        if suffix=='r':
            lstm_output = lstm_output[::-1]
        return self._dropout(lstm_output, dropout_prob)
    def _q_z_x(self, X, mask = None, dropout_prob = 0., anneal =1.):
        """
        Inference
        X: nbatch x time x dim_observations 
        Returns: z_q (nbatch x time x dim_stochastic), mu_q (nbatch x time x dim_stochastic) and cov_q (nbatch x time x dim_stochastic)
        """
        self._p('Building with RNN dropout:'+str(dropout_prob))
        embedding         = self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], X)
        h_r               = self._LSTM_RNN_layer(embedding, 'r', temporalMask = mask, dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])
        if self.params['inference_model']=='LR':
            h_l           = self._LSTM_RNN_layer(embedding, 'l', temporalMask = mask, dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])
            hidden_state  = (h_r+h_l)/2.
        elif self.params['inference_model']=='R':
            hidden_state  = h_r
        else:
            raise ValueError('Bad inference model')
        z_q, mu_q, cov_q  = self._aggregateLSTM(hidden_state)
        return z_q, mu_q, cov_q
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

    """
    Resetting the datasets stored on the GPU
    """
    def resetDataset(self, dataset, quiet=False):
        if not quiet:
            dimlist = self.dimData()
            dim_str = ','.join([str(d) for d in dimlist])
            self._p('Original dim: '+dim_str)
        newX, newM = dataset['tensor'].astype(config.floatX), dataset['mask'].astype(config.floatX)
        self.setData(newX=newX, newMask = newM)
        if not quiet:
            dimlist = self.dimData()
            dim_str = ','.join([str(d) for d in dimlist])
            self._p('New dim: '+dim_str)

    """ Building Model """
    def _buildModel(self):
        """ 
        This function builds high level function to build and setup theano functions: 
        """
        idx                = T.vector('idx',dtype='int64')
        """ Setup tags for debugging """
        idx_tag            = np.array([0,1]).astype('int64')
        idx.tag.test_value = idx_tag

        X_init             = np.random.uniform(0,1,size=(3,5,self.params['dim_observations'])).astype(config.floatX)
        M_init             = np.ones_like(X_init[:,:,0]).astype(config.floatX) 
        M_init[0,4:] = 0.
        M_init[1,1:] = 0.
        M_init[2,2:] = 0.

        self.dataset       = theano.shared(X_init, name = 'dataset')
        self.mask          = theano.shared(M_init, name = 'mask')
        self.dataset.tag.test_value = X_init
        self.mask.tag.test_value    = M_init

        X                  = self.dataset[idx]
        M                  = self.mask[idx]
        newX, newMask      = T.tensor3('newX',dtype=config.floatX), T.matrix('newMask',dtype=config.floatX)
        inputs_set         = [newX, newMask]
        updates_set        = [(self.dataset,newX), (self.mask,newMask)]
        outputs_dim        = [self.dataset.shape, self.mask.shape]  
        self.setData       = theano.function(inputs_set, None, updates=updates_set)
        self.dimData       = theano.function([],outputs_dim)

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
            train_cost = self._neg_elbo(X, M, anneal = anneal, dropout_prob = self.params['rnn_dropout'], additional=traindict)
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
            self.train_debug = theano.function(fxn_inputs,[train_cost,norm_list[0],norm_list[1],
                norm_list[2], traindict['nll'], traindict['kl'], anneal.sum()], updates = optimizer_up, name='Train (with Debug Outputs)')
        #Updates ack
        self.updates_ack         = True
        """ Setup functions to evaluate the model """
	evaldict                 = {}
        eval_cost                = self._neg_elbo(X, M, anneal = 1., dropout_prob = 0., additional= evaldict)
        self.evaluate            = theano.function([idx], eval_cost, name = 'Evaluate Bound',allow_input_downcast=True)
        self.nll_feat            = theano.function([idx],evaldict['nll_mat'], name = 'NLL features',allow_input_downcast=True)
        self.nll_batch           = theano.function([idx],[evaldict['hid_out'],evaldict['nll_mat']], name = 'NLL batch',allow_input_downcast=True)
        self.posterior_inference = theano.function([idx], [evaldict['z_q'], evaldict['mu_q'], evaldict['cov_q']], name='Posterior Inference',allow_input_downcast=True)
        if self.params['data_type']=='binary':
            self.emission_fxn        = theano.function([evaldict['z_q']], [evaldict['bin_prob']] , name='Emission',allow_input_downcast=True)
        else:
            self.emission_fxn        = theano.function([evaldict['z_q']], [evaldict['real_mu'], evaldict['real_logcov']] , name='Emission',allow_input_downcast=True)
        evaldict['z_q'].name = 'Z'
        self.transition_fxn = theano.function([evaldict['z_q']], [evaldict['mu_t'], evaldict['cov_t']], name='Transition Function',allow_input_downcast=True)
        self.posterior_inference_data = theano.function([X,M], [evaldict['z_q'], evaldict['mu_q'], evaldict['cov_q']], name='Posterior Inference',allow_input_downcast=True)
        self._p('Completed DMM setup')
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
if __name__=='__main__':
    """ use this to check compilation for various options"""
    from parse_args import params
    params['data_type']         = 'binary'
    params['dim_observations']  = 88 
    dmm = DMM(params, paramFile = 'tmp')
    os.unlink('tmp')
    import ipdb;ipdb.set_trace()
