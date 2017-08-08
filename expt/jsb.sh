THEANO_FLAGS="gpuarray.preallocate=1.,scan.allow_gc=False,compiledir_format=gpu3" python2.7 train.py -vm R -infm structured -ar 5000 -dset jsb -uid DKF-ar
