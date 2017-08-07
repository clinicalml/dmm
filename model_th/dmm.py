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
    """ Initialize parameters for inference and generative model """
    def _createParams(self):
        """ Model parameters """
        npWeights = OrderedDict()
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights
    def _createGenerativeParams(self, npWeights):
        """ Create weights/params for generative model """
        npWeights['B_mu_W']  = self._getWeight((self.params['dim_baselines'], self.params['dim_stochastic']))
        npWeights['B_cov_W'] = self._getWeight((self.params['dim_baselines'], self.params['dim_stochastic']))
        DIM_HIDDEN           = self.params['dim_hidden']
        DIM_STOCHASTIC       = self.params['dim_stochastic']
        DIM_HIDDEN_TRANS     = DIM_HIDDEN

        """ Transition Function """
        for l in range(self.params['transition_layers']):
            dim_input, dim_input_act, dim_output = DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS
            if l==0:
                dim_input     = self.params['dim_stochastic']
            npWeights['p_trans_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_trans_b_'+str(l)] = self._getWeight((dim_output,))
        MU_COV_INP, MU_COV_INP_ACT= DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS*2
        if self.params['transition_layers']==0:
            MU_COV_INP     = self.params['dim_stochastic']
        npWeights['p_trans_W_mu']  = 0.1*self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_b_mu']  = 0.1*self._getWeight((self.params['dim_stochastic'],))
        npWeights['p_trans_W_cov'] = 0.1*self._getWeight((MU_COV_INP,self.params['dim_stochastic']))
        npWeights['p_trans_b_cov'] = 0.1*self._getWeight((self.params['dim_stochastic'],))

        """ Emission Function parameters """
        for l in range(self.params['emission_layers']):
            dim_input,dim_output  = DIM_HIDDEN, DIM_HIDDEN
            if l==0:
                dim_input = self.params['dim_stochastic']
            npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
        if self.params['data_type']=='mixed':
            dim_out = np.where(self.params['feature_types']=='binary')[0].shape[0] + 2*np.where(self.params['feature_types']=='continuous')[0].shape[0]
        elif self.params['data_type'] == 'real':
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

        #Embed the Input -> RNN_SIZE
        dim_input, dim_output= DIM_INPUT, RNN_SIZE
        npWeights['q_W_input_0'] = self._getWeight((dim_input, dim_output))
        npWeights['q_b_input_0'] = self._getWeight((dim_output,))

        #Setup weights for LSTM
        self._createLSTMWeights(npWeights)

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
                npWeights['W_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE*4))
                npWeights['b_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE*4,), scheme='lstm')
                npWeights['U_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE*4),scheme='lstm')
            elif self.params['rnn_cell']=='rnn':
                npWeights['W_rnn_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE))
                npWeights['b_rnn_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,), scheme='lstm')
                npWeights['U_rnn_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE),scheme='orthogonal')
            else:
                raise ValueError('Invalid setting for RNN cell')

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    """
    Generative Model 
    """
    def _transition(self, z, U=None, trans_params = None, anneal = 1.):
        """
        Transition Function for DMM
        Input:  z [bs x T x dim]
        Output: mu/cov [bs x T x dim]
        """
        if trans_params is None:
            trans_params = self.tWeights
        hid    = z
        for l in range(self.params['transition_layers']):
            hid= self._LinearNL(trans_params['p_trans_W_'+str(l)],trans_params['p_trans_b_'+str(l)],hid)
        if U is None:
            mu     = T.dot(hid, trans_params['p_trans_W_mu']) + trans_params['p_trans_b_mu']
            cov    = T.nnet.softplus(T.dot(hid, trans_params['p_trans_W_cov'])+trans_params['p_trans_b_cov'])
            return mu, cov
        else:
            hid_u     = U
            for l in range(self.params['transition_layers']):
                hid_u = self._LinearNL(trans_params['p_trans_W_act_'+str(l)], trans_params['p_trans_b_act_'+str(l)], hid_u) 
            hid_r  = T.concatenate([anneal*hid, hid_u], axis=-1)
            mu     = T.dot(hid_r, trans_params['p_trans_W_act_mu']) + trans_params['p_trans_b_act_mu']
            cov    = T.nnet.softplus(T.dot(hid_r, trans_params['p_trans_W_act_cov'])+trans_params['p_trans_b_act_cov'])
            return mu, cov
        """
        mu     = T.dot(hid, trans_params['p_trans_W_mu']) + trans_params['p_trans_b_mu']
        cov    = T.nnet.softplus(T.dot(hid, trans_params['p_trans_W_cov'])+trans_params['p_trans_b_cov'])
        if U is None:
            return mu, cov
        else:
            hid_u     = U
            for l in range(self.params['transition_layers']):
                hid_u = self._LinearNL(trans_params['p_trans_W_act_'+str(l)], trans_params['p_trans_b_act_'+str(l)], hid_u) 
            mu_u  = T.dot(hid_u, trans_params['p_trans_W_act_mu'])+trans_params['p_trans_b_act_mu']
            cov_u = T.nnet.softplus(T.dot(hid_u, trans_params['p_trans_W_act_cov'])+trans_params['p_trans_b_act_cov'])
            mu_f, cov_f = self._pog(mu_u, cov_u, mu, cov)
            return mu_f, cov_f
        """

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
    def _temporalKL(self, mu_q, cov_q, mu_prior, cov_prior, maxTmask):
        """
        KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q)
                        + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        """
        assert np.all(cov_q.tag.test_value>0.),'should be positive'
        assert np.all(cov_prior.tag.test_value>0.),'should be positive'
        diff_mu = mu_prior-mu_q
        KL      = T.log(cov_prior)-T.log(cov_q) - 1. + cov_q/cov_prior + diff_mu**2/cov_prior
        KL_t    = 0.5*KL.sum(2)
        KLmasked  = (KL_t*maxTmask)
        return KLmasked.sum()

    def _neg_elbo(self, X, B, M, maxTmask, U = None, anneal = 1., dropout_prob = 0., additional = None):
        z_q, mu_q, cov_q   = self._q_z_x(X, U= U, B=B, maxTmask = maxTmask, dropout_prob = dropout_prob, anneal = anneal)
        ##added fix for NaN - this should work but try something simpler first
        #z_q_m              = z_q*maxTmask[:,:,None]   + T.zeros_like(z_q)*(1-maxTmask[:,:,None])
        #mu_q_m             = mu_q*maxTmask[:,:,None]  + T.zeros_like(mu_q)*(1-maxTmask[:,:,None])
        #cov_q_m            = cov_q*maxTmask[:,:,None] + T.ones_like(cov_q)*(1-maxTmask[:,:,None])
        mu_trans, cov_trans= self._transition(z_q, U = U, anneal = anneal)
        #mu_trans_m         = mu_trans*maxTmask[:,:,None]  + T.zeros_like(mu_trans)*(1-maxTmask[:,:,None])
        #cov_trans_m        = cov_trans*maxTmask[:,:,None] + T.ones_like(cov_trans)*(1-maxTmask[:,:,None])
        """ Initial prior distribution """
        B_mu               = T.dot(B, self.tWeights['B_mu_W'])[:,None,:] 
        B_cov              = T.nnet.softplus(T.dot(B, self.tWeights['B_cov_W']))[:,None,:]
        if self.params['fixed_init_prior']:
            B_mu = B_mu*0.
            B_cov= B_cov*0.+1.
        mu_prior           = T.concatenate([B_mu, mu_trans[:,:-1,:]], axis=1)
        cov_prior          = T.concatenate([B_cov,cov_trans[:,:-1,:]],axis=1)
        """
        Regularization using 
        simulation during learning 
        """
        #if dropout_prob>0:
        #    eps_prior      = self.srng.normal(size=(mu_prior.shape))
        #    z_sim          = mu_prior + T.sqrt(cov_prior)*eps_prior
        #    hid_out_sim    = self._emission(z_sim)
        #    nll_mat_sim    = self._nll_mixed(hid_out_sim, X, mask=M, ftypes = self.params['feature_types'])
        #    sim_reg        = (1-anneal)*nll_mat_sim.sum()
        KL                 = self._temporalKL(mu_q, cov_q, mu_prior, cov_prior, maxTmask = maxTmask)
        hid_out            = self._emission(z_q)
	params             = {}
        if self.params['data_type'] == 'mixed':
            #Set the logcovariances to a small fixed value
            #cont_idx       = len(np.where(self.params['feature_types']=='continuous')[0])
            #mask_cont      = T.set_subtensor(T.ones_like(hid_out)[:,:,-cont_idx:],0)
            #hid_out        = hid_out*(mask_cont) + np.log(0.1)*(1-mask_cont)
            nll_mat        = self._nll_mixed(hid_out, X, mask=M, ftypes = self.params['feature_types'], params = params)
        elif self.params['data_type'] == 'binary':
            nll_mat        = self._nll_binary(hid_out, X, mask = M, params= params)
        elif self.params['data_type']=='real':
            dim_obs        = self.params['dim_observations']
            nll_mat        = self._nll_gaussian(hid_out[:,:,:dim_obs], hid_out[:,:,dim_obs:dim_obs*2], X, mask = M, params=params)
        else:
            raise ValueError('Invalid Data Type'+str(self.params['data_type']))
        #if dropout_prob>0:
        #    nll            = nll_mat.sum()+sim_reg
        #else:
        nll            = nll_mat.sum()
        #Evaluate negative ELBO
        neg_elbo       = nll+KL
        if additional is not None:
            additional['hid_out']= hid_out
            additional['nll_mat']= nll_mat
            additional['nll_batch']= nll_mat.sum((1,2))
            additional['nll_feat']= nll_mat.sum((0,2))
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
    def _pog(self, mu_1,cov_1, mu_2,cov_2):
        cov_f = T.sqrt((cov_1*cov_2)/(cov_1+cov_2))
        mu_f  = (mu_1*cov_2+mu_2*cov_1)/(cov_1+cov_2) 
        return mu_f, cov_f

    def _aggregateLSTM(self, hidden_state, B = None, anneal = 1.):
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
            mu_trans, cov_trans = self._transition(z_prev, trans_params = tParams, anneal = anneal)
            mu_t       = T.dot(h_t,tParams['q_W_mu'])+tParams['q_b_mu']
            cov_t      = T.nnet.softplus(T.dot(h_t, tParams['q_W_cov'])+tParams['q_b_cov'])
            mu_f, cov_f= self._pog(mu_trans, cov_trans, mu_t, cov_t)
            z_f        = mu_f+T.sqrt(cov_f)*eps_t
            return z_f, mu_f, cov_f
        # eps: [T x bs x dim_stochastic]
        eps         = self.srng.normal(size=(hidden_state.shape[0],hidden_state.shape[1],self.params['dim_stochastic']))
        B_mu        = T.dot(B, self.tWeights['B_mu_W'])
        B_cov       = T.nnet.softplus(T.dot(B, self.tWeights['B_cov_W']))
        z0          = B_mu + T.sqrt(B_cov)*self.srng.normal(size=B_mu.shape)
        #else:
        #    z0          = T.zeros((eps.shape[1], self.params['dim_stochastic']))
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


    def _aggregateLSTM_U(self, hidden_state, U=None, B = None, anneal = 1.):
        assert U is not None,'expecting U'
        if self.params['dim_stochastic']==1:
            raise ValueError('dim_stochastic must be larger than 1 dimension')
        """
        Use the true prior
            * Gradients for predicting z_2 will be propagated 
        """
        def st_true(h_t, eps_t, u_t, z_prev, *params):
            tParams = {}
            for p in params:
                tParams[p.name] = p
            mu_trans, cov_trans = self._transition(z_prev, U=u_t, trans_params=tParams, anneal = anneal)
            mu_t       = T.dot(h_t,tParams['q_W_mu'])+tParams['q_b_mu']
            cov_t      = T.nnet.softplus(T.dot(h_t,tParams['q_W_cov'])+tParams['q_b_cov'])
            #POG Approximation
            mu_f, cov_f= self._pog(mu_trans, cov_trans, mu_t, cov_t) 
            z_f        = mu_f+T.sqrt(cov_f)*eps_t
            return z_f, mu_f, cov_f
        def st_approx(h_t, eps_t, u_t, z_prev, *params):
            tParams = {}
            for p in params:
                tParams[p.name] = p
            h_z        = T.tanh(T.dot(z_prev,tParams['q_W_st'])+tParams['q_b_st'])
            h_act      = T.tanh(T.dot(u_t,tParams['q_W_act'])+tParams['q_b_act'])
            h_next     = (1./3.)*(h_t+h_z+h_act)
            mu_t       = T.dot(h_next,tParams['q_W_mu'])+tParams['q_b_mu']
            cov_t      = T.nnet.softplus(T.dot(h_next,tParams['q_W_cov'])+tParams['q_b_cov'])
            z_t        = mu_t+T.sqrt(cov_t)*eps_t
            return z_t, mu_t, cov_t
        # eps: [T x bs x dim_stochastic]
        eps         = self.srng.normal(size=(hidden_state.shape[0],hidden_state.shape[1],self.params['dim_stochastic']))
        B_mu        = T.dot(B, self.tWeights['B_mu_W'])
        B_cov       = T.nnet.softplus(T.dot(B, self.tWeights['B_cov_W']))
        z0          = B_mu + T.sqrt(B_cov)*self.srng.normal(size=B_mu.shape)
        #z0          = T.zeros((eps.shape[1], self.params['dim_stochastic']))
        if self.params['use_generative_prior']=='true':
            non_seq = [self.tWeights[k] for k in ['q_W_mu','q_b_mu','q_W_cov','q_b_cov']]+[self.tWeights[k] for k in self.tWeights if '_trans_' in k]
            scan_fxn=st_true
        elif self.params['use_generative_prior']=='approx':
            non_seq = [self.tWeights[k] for k in ['q_W_st', 'q_b_st', 'q_W_mu','q_b_mu','q_W_cov','q_b_cov','q_W_act','q_b_act']]
            scan_fxn=st_approx 
        else:
            raise NotImplemented('Should not reach here')
        rval, _     = theano.scan(scan_fxn, sequences=[hidden_state, eps, U.swapaxes(0,1)],
                                    outputs_info=[z0, None,None],
                                    non_sequences=non_seq,
                                    name='structuredApproximation')
        z, mu, cov  = rval[0].swapaxes(0,1), rval[1].swapaxes(0,1), rval[2].swapaxes(0,1)
        return z, mu, cov

    def _LSTM_RNN_layer(self, inp, suffix, temporalMask = None, dropout_prob=0., RNN_SIZE = None, init_h = None):
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
            #step contains atleast one observed feature 
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

        lstm_embed = T.dot(inp.swapaxes(0,1),self.tWeights['W_'+rnn_cell+'_'+suffix+'_0'])+ self.tWeights['b_'+rnn_cell+'_'+suffix+'_0']
        nsteps     = lstm_embed.shape[0]
        n_samples  = lstm_embed.shape[1]
        if self.params['rnn_cell']=='lstm':
            o_info = [T.zeros((n_samples,RNN_SIZE)), T.ones((n_samples,RNN_SIZE))]
        else:
            o_info = [T.zeros((n_samples,RNN_SIZE))]
        if init_h is not None:
            if self.params['rnn_cell']=='lstm':
                o_info = [init_h, T.ones((n_samples,RNN_SIZE))]
            else:
                o_info = [init_h]
        n_seq      = [self.tWeights['U_'+rnn_cell+'_'+suffix+'_0']]
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
    def _q_z_x(self, X, U = None, B = None, maxTmask = None, dropout_prob = 0., anneal =1.):
        """
        Inference
        X: nbatch x time x dim_observations 
        Returns: z_q (nbatch x time x dim_stochastic), mu_q (nbatch x time x dim_stochastic) and cov_q (nbatch x time x dim_stochastic)
        """
        self._p('Building with RNN dropout:'+str(dropout_prob))
        embedding         = self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], X)
        h_r               = self._LSTM_RNN_layer(embedding, 'r', temporalMask = maxTmask, dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])
        if self.params['inference_model']=='LR':
            h_l           = self._LSTM_RNN_layer(embedding, 'l', temporalMask = maxTmask, dropout_prob = dropout_prob, RNN_SIZE = self.params['rnn_size'])
            hidden_state  = (h_r+h_l)/2.
        elif self.params['inference_model']=='R':
            hidden_state  = h_r
        else:
            raise ValueError('Bad inference model')
        z_q, mu_q, cov_q  = self._aggregateLSTM(hidden_state, B = B, anneal = anneal)
        return z_q, mu_q, cov_q
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

    """
    Resetting Datasets
    """
    def _setupTmask(self, mask_tensor):
        p_obs      = (mask_tensor.sum(-1)>0.)*1.
        maxt       = p_obs.shape[1]-np.argmax(p_obs[:,::-1],1)
        result = []
        for p, midx in zip(np.zeros_like(p_obs),maxt):
            p[:midx] = 1.
            result.append(p)
        maxTmask   = np.array(result).astype(config.floatX)
        return maxTmask

    def resetDataset(self, dataset, quiet=False):
        maxTmask = self._setupTmask(dataset['features']['obs_tensor'])
        if not quiet:
            dimlist = self.dimData()
            dim_str = ','.join([str(d) for d in dimlist])
            self._p('Original dim: '+dim_str)
        newX, newM = dataset['features']['tensor'].astype(config.floatX), dataset['features']['obs_tensor'].astype(config.floatX)
        newB       = dataset['baselines']['tensor'].astype(config.floatX)
        self.setData(newX=newX, newMask = newM, newB = newB, newTmask = maxTmask)
        if not quiet:
            dimlist = self.dimData()
            dim_str = ','.join([str(d) for d in dimlist])
            self._p('New dim: '+dim_str)

    """ Building Model """
    def _buildModel(self):
        """ High level function to build and setup theano functions """
        idx                = T.vector('idx',dtype='int64')
        idx.tag.test_value = np.array([0,1]).astype('int64')
        X_init             = np.random.uniform(0,1,size=(3,5,self.params['dim_observations'])).astype(config.floatX)
        """
        Setup tags
        """
        M_init             = ((X_init>0.5)*1.).astype(config.floatX) 
        M_init[0,4:,:] = 0.
        M_init[0,1,:] = 0.
        M_init[1,2:,:] = 0.
        maxT_init          = self._setupTmask(M_init)
        
        B_init             = np.random.uniform(0,1,size=(3,self.params['dim_baselines'])).astype(config.floatX)
        self.dataset       = theano.shared(X_init, name = 'X')
        self.dataset_b     = theano.shared(B_init, name = 'B')
        self.mask          = theano.shared(M_init, name = 'M')
        self.maskT         = theano.shared(maxT_init, name = 'maskT')

        X                = self.dataset[idx]
        B                = self.dataset_b[idx]
        M                = self.mask[idx]
        maxTmask         = self.maskT[idx]

        newX, newMask, newB= T.tensor3('newX',dtype=config.floatX), T.tensor3('newMask',dtype=config.floatX), T.matrix('newB',dtype=config.floatX)
        newTmask           = T.matrix('newTmask',dtype=config.floatX)
        inputs_set         = [newX, newMask, newB, newTmask] 
        updates_set        = [(self.dataset,newX), (self.mask,newMask), (self.dataset_b, newB), (self.maskT, newTmask)]
        outputs_dim        = [self.dataset.shape, self.mask.shape, self.dataset_b.shape, self.maskT.shape]  
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
            train_cost = self._neg_elbo(X, B, M, maxTmask, U = U, anneal = anneal, dropout_prob = self.params['rnn_dropout'], additional=traindict)
            if self.params['time_to_death']:
                traindict
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
        eval_cost                = self._neg_elbo(X, B, M, maxTmask, U = U, anneal = 1., dropout_prob = 0., additional= evaldict)
        self.evaluate            = theano.function([idx], eval_cost, name = 'Evaluate Bound',allow_input_downcast=True)
        self.nll_feat            = theano.function([idx],evaldict['nll_mat'], name = 'NLL features',allow_input_downcast=True)
        self.nll_batch           = theano.function([idx],[evaldict['hid_out'],evaldict['nll_mat']], name = 'NLL batch',allow_input_downcast=True)
        self.posterior_inference = theano.function([idx], [evaldict['z_q'], evaldict['mu_q'], evaldict['cov_q']], name='Posterior Inference',allow_input_downcast=True)
        self.emission_fxn        = theano.function([evaldict['z_q']], [evaldict['bin_prob'], evaldict['real_mu'], evaldict['real_logcov']] , name='Emission',allow_input_downcast=True)
	self.init_prior          = theano.function([B],[evaldict['b_mu'],evaldict['b_cov']], name = 'Initial Prior',allow_input_downcast=True)
        evaldict['z_q'].name     = 'Z'
        X.name = 'X'
        B.name = 'B'
        maxTmask.name = 'maxT'
        self.transition_fxn = theano.function([evaldict['z_q']], [evaldict['mu_t'], evaldict['cov_t']], name='Transition Function',allow_input_downcast=True)
        self.posterior_inference_data = theano.function([X,B,maxTmask], [evaldict['z_q'], evaldict['mu_q'], 
                evaldict['cov_q']], name='Posterior Inference',allow_input_downcast=True)
        self._p('Completed DMM setup')
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
if __name__=='__main__':
    """ use this to check compilation for various options"""
    from parse_args import params
    params['data_type']         = 'binary'
    params['dim_observations']  = 10
    dmm = DMM(params, paramFile = 'tmp')
    os.unlink('tmp')
    import ipdb;ipdb.set_trace()
