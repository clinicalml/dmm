THEANO_FLAGS="gpuarray.preallocate=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train.py -vm R -infm structured -ar 5000 -dset nottingham -uid DKF-ar
