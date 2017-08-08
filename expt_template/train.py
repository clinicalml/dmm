import os,time,sys
import fcntl,errno
import socket
sys.path.append('../')
from dmm_data.load import load
from parse_args import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError

dataset = load('synthetic')
params['savedir']+='-synthetic'
createIfAbsent(params['savedir'])

for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

""" Import files for learning """
start_time = time.time()
from   model_th.dmm import DMM
import model_th.learning as DMM_learn
import model_th.evaluate as DMM_evaluate
displayTime('import DMM',start_time, time.time())
dmm = None

"""
Build new model
"""
start_time = time.time()
pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
print 'Training model from scratch. Parameters in: ',pfile
dmm  = DMM(params, paramFile = pfile)
displayTime('Building dmm',start_time, time.time())

"""
Savefile where model checkpoints will be saved
"""
savef     = os.path.join(params['savedir'],params['unique_id']) 
print 'Savefile: ',savef
start_time= time.time()

""" Training loop """
savedata = DMM_learn.learn(dmm, dataset['train'],
                                epoch_start =0 , 
                                epoch_end = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval=dataset['valid'],
                                shuffle    = True
                                )
displayTime('Running DMM',start_time, time.time()         )

dmm = None

""" Reload the best DMM based on the validation error """
epochMin, valMin, idxMin = getLowestError(savedata['valid_bound'])
reloadFile= pfile.replace('-config.pkl','')+'-EP'+str(int(epochMin))+'-params.npz'

print 'Loading from : ',reloadFile
params['validate_only']          = True
dmm_best                         = DMM(params, paramFile = pfile, reloadFile = reloadFile)

"""
Evaluate on the test set
"""
additional                       = {}
savedata['bound_test']      = DMM_evaluate.evaluateBound(dmm_best,  dataset['test'], batch_size = params['batch_size'])
saveHDF5(savef+'-final.h5',savedata)
print 'Experiment Name: <',params['expt_name'],'> Test Bound: ',savedata['bound_test']
