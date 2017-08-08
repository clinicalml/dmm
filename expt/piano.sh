THEANO_FLAGS="gpuarray.preallocate=0.6,scan.allow_gc=False,compiledir_format=gpu3" python2.7 train.py -infm R -ar 2000 -dset piano -uid DKF-ar
