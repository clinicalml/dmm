from datasets.load import loadDataset 

def loadSyntheticData():
    dataset = {}
    return dataset

def load(dset):
    if dset   in ['jsb','nottingham','musedata','piano']:
        musicdata = loadDataset(dset)
        dataset   = {}
        for k in ['train','valid','test']:
            dataset[k] = {}  
            dataset[k]['tensor'] = musicdata[k] 
            dataset[k]['mask']   = musicdata['mask_'+k]
        dataset['data_type']        = musicdata['data_type']
        dataset['dim_observations'] = musicdata['dim_observations']
    elif dset == 'synthetic':
        dataset = loadSyntheticData()
    else:
        raise ValueError('Invalid dataset: '+dset)
    return dataset

if __name__=='__main__':
    data = load('jsb')
    import ipdb; ipdb.set_trace()
