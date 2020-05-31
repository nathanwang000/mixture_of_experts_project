'''
utility functions for dataset related operations
'''
import glob, os
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

class FileDataset(Dataset):
    '''
    given a directory path, read in the pkl files in the directory
    '''
    def __init__(self, path):
        self.path = path
        self.length = len(glob.glob(os.path.join(path, "*.pkl")))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (joblib.load(os.path.join(self.path, "{}.pkl".format(idx))),)
    
class ColumnDataset(Dataset):
    ''' 
    given datasets, treat each dataset as a column
    e.g ColumnDataset(d1, d2, ...)[idx]
    would give (d1[idx], d2[idx], ...)
    '''
    def __init__(self, *datasets):
        '''
        Args:
            datasets: datasets to be used as columns; assumes each dataset gives a tuple of items
        '''
        self.datasets = datasets
        self.length = min([len(dataset) for dataset in datasets])
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return sum([dataset[idx] for dataset in self.datasets], ())

class MergeDataset(Dataset):
    ''' 
    given a dataset that gives tuples of items, merge the items
    e.g MergeDataset(ColumnDataset(d1, d2))
    would give (torch.concat(d1[idx], d2[idx]), )
    '''
    def __init__(self, dataset, axis=0):
        '''
        Args:
            datasets: datasets to be used as columns; assumes each dataset gives a tuple of items
        '''
        self.dataset = dataset
        self.axis = axis
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset[idx][0].ndim == 0:
            return (torch.tensor(self.dataset[idx]), )
        return (torch.cat(self.dataset[idx], axis=self.axis),)

def train_val_test_split(*datasets, **kwargs):
    """ 
    Return stratified split of datasets, stratified by stratify
    Args:
        datasets (pytorch datasets): each dataset should have the same length 
        kwargs: would use "stratify" (np array), and "train_val_random_seed"
    Returns:
        train, val, test dataset for each dataset in datasets
    """
    stratify = kwargs.get('stratify', None)
    train_val_random_seed = kwargs.get('train_val_random_seed', 0)

    indices = np.arange(len(datasets[0]))
    idx_train_val, idx_test = train_test_split(indices, stratify=stratify, test_size=0.2,
                                               random_state=train_val_random_seed)
    if stratify is not None:
        stratify = stratify[idx_train_val]
    idx_train, idx_val = train_test_split(idx_train_val, stratify=stratify, test_size=0.125,
                                          random_state=train_val_random_seed)

    # apply on datasets
    res = []
    for dataset in datasets:
        res.extend([
            Subset(dataset, idx_train),
            Subset(dataset, idx_val),
            Subset(dataset, idx_test),            
        ])

    # for use of split train data
    # joblib.dump(idx_train, 'train_idx.pkl')
    return res

def dataset2numpy(dataset):
    '''
    assume the dataset is not too large; otherwise numpy will be as large
    return numpy representation of the first item
    '''
    return next(iter(DataLoader(dataset, batch_size=len(dataset))))[0].numpy()

