"""
Rahul G. Krishnan

Script to setup experiments either on HPC or individually 
"""
import numpy as np
from collections import OrderedDict
import argparse,os

parser = argparse.ArgumentParser(description='Setup Expts')
parser.add_argument('-hpc','--onHPC',action='store_true') 
parser.add_argument('-dset','--dataset', default='jsb',action='store')
parser.add_argument('-ngpu','--num_gpus', default=4,action='store',type=int)
args = parser.parse_args()

#MAIN FLAGS
onHPC        = args.onHPC
DATASET      = args.dataset 
THFLAGS      = 'THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu<rand_idx>" ' 

#Get dataset
dataset      = DATASET.split('-')[0]
all_datasets = ['jsb','piano','nottingham','musedata']
assert dataset in all_datasets,'Dset not found: '+dataset
all_expts    = OrderedDict()
for dset in all_datasets:
    all_expts[dset] = OrderedDict()

#Experiments to run for each dataset
all_expts['jsb']['DKF-ar']        ='python2.7 train.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['nottingham']['DKF-ar'] ='python2.7 train.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['musedata']['DKF-ar']   ='python2.7 train.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['piano']['DKF-ar']      ='python2.7 train.py -vm R -infm structured -ar 5000 -dset <dataset>' 

if onHPC:
    DIR = './hpc_'+dataset
    os.system('rm -rf '+DIR) 
    os.system('mkdir -p '+DIR)
    with open('template.q') as ff:
        template = ff.read()
    runallcmd    = ''
    for name in all_expts[dataset]:
        runcmd  = all_expts[dataset][name].replace('<dataset>',DATASET)+' -uid '+name
        command = THFLAGS.replace('<rand_idx>',str(np.random.randint(args.num_gpus)))+runcmd
        with open(DIR+'/'+name+'.q','w') as f:
            f.write(template.replace('<name>',name).replace('<command>',command))
        print 'Wrote to:',DIR+'/'+name+'.q'
        runallcmd+= 'qsub '+name+'.q\n'
    with open(DIR+'/runall.sh','w') as f:
        f.write(runallcmd)
else:
    for name in all_expts[dataset]:
        runcmd  = all_expts[dataset][name].replace('<dataset>',DATASET)+' -uid '+name
        command = THFLAGS.replace('<rand_idx>',str(np.random.randint(args.num_gpus)))+runcmd
        print command
