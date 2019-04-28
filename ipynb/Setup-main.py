#General Purpose Imports
import numpy as np
import glob, os, sys, time
sys.path.append('../')
from utils.misc import getConfigFile, readPickle, displayTime

#Import load function to load synthetic data
from dmm_data.load import load
dataset = load('synthetic')
print type(dataset), dataset.keys()

print 'Dimensionality of the observations: ', dataset['dim_observations']
print 'Data type of features:', dataset['data_type']
for dtype in ['train','valid','test']:
    print 'dtype: ',dtype, ' type(dataset[dtype]): ',type(dataset[dtype])
    print [(k,type(dataset[dtype][k]), dataset[dtype][k].shape) for k in dataset[dtype]]
    print '--------\n'

start_time = time.time()
from   model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate
displayTime('importing DMM',start_time, time.time())

params = readPickle('../default.pkl')[0]
for k in params:
    print k, '\t',params[k]
params['data_type'] = dataset['data_type']
params['dim_observations'] = dataset['dim_observations']

#The dataset is small, lets change some of the default parameters and the unique ID
params['dim_stochastic'] = 2
params['dim_hidden']     = 40
params['rnn_size']       = 80
params['epochs']         = 40
params['batch_size']     = 200
params['unique_id'] = params['unique_id'].replace('ds-100','ds-2').replace('dh-200','dh-40').replace('rs-600','rs-80')
params['unique_id'] = params['unique_id'].replace('ep-2000','ep-40').replace('bs-20','bs-200')

#Create a temporary directory to save checkpoints
params['savedir']   = params['savedir']+'-ipython/'
os.system('mkdir -p '+params['savedir'])

#Specify the file where `params` corresponding for this choice of model and data will be saved
pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'

print 'Checkpoint prefix: ', pfile
dmm  = DMM(params, paramFile = pfile)