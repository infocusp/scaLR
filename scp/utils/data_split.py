import anndata as ad
from anndata import AnnData
from sklearn.model_selection import GroupShuffleSplit
import json
from .file import write_data, dump_json, read_data
import os

def generate_split(datapath:str, split_ratio:list[float], target:str, stratify:str=None, path:str=None) -> dict:
    """Generate a list of indices for train/val/test split of whole dataset

    Args:
        datapath: path to full data
        split_ratio: ratio to split number of samples in
        target: target for classification present in `obs`.
        stratify: optional parameter to stratify the split upon parameter.
        path: path to store generated split in json format/
    
    Returns:
        dict with 'train', 'test' and 'val' indices list.
    
    """
    
    adata = read_data(datapath)
    metadata = adata.obs
    metadata['inds'] = range(len(metadata))
    n_cls = len(metadata[target].unique())

    total_ = sum(split_ratio)
    train_ratio = split_ratio[0]/total_
    val_ratio = split_ratio[1]/total_
    test_ratio = split_ratio[2]/total_
    
    test_splitter = GroupShuffleSplit(test_size=test_ratio, n_splits=10, random_state=42)
    training_inds, testing_inds = next(test_splitter.split(metadata, groups = metadata[stratify] if stratify is not None else None))

    if len(metadata[target].iloc[testing_inds].unique()) != n_cls:
        print('WARNING: All classes are not present in Test set')

    train_data = metadata.iloc[training_inds]

    val_splitter = GroupShuffleSplit(test_size=val_ratio/(val_ratio+train_ratio), n_splits=10, random_state=42)
    fake_train_inds, fake_val_inds = next(val_splitter.split(train_data, groups = train_data[stratify] if stratify is not None else None))

    true_test_inds = testing_inds.tolist()
    true_val_inds = train_data.iloc[fake_val_inds]['inds'].tolist()
    true_train_inds = train_data.iloc[fake_train_inds]['inds'].tolist()

    if len(metadata[target].iloc[true_val_inds].unique()) != n_cls:
        print('WARNING: All classes are not present in Validation set')

    if len(metadata[target].iloc[true_train_inds].unique()) != n_cls:
        print('WARNING: All classes are not present in Train set')

    assert len(set(true_test_inds).intersection(true_train_inds)) == 0
    assert len(set(true_val_inds).intersection(true_train_inds)) == 0
    assert len(set(true_val_inds).intersection(true_test_inds)) == 0

    print('Length of train set: ', len(true_train_inds))
    print('Length of val set: ', len(true_val_inds))
    print('Length of test set: ', len(true_test_inds), flush=True)
    
    data_split = {
        'train':true_train_inds,
        'val':true_val_inds,
        'test':true_test_inds
    }

    if path is not None:
        dump_json(data_split, path)

    return data_split

def split_data(datapath:str, data_split:dict, dirpath:str, chunksize:int=None):
    """Split the full data based upon generated indices lists and write it to disk.
    
    Args:
        datapath: path to full dataset
        data_split: dict containing list of indices for train/val/test splits
        dirpath: path to store new split data.
        chunksize: number of samples to store in one chunk, after splitting the data.

    """
    
    dstype = ['train','val','test']
    for typ in dstype:
        if chunksize is None:
            adata = read_data(datapath)
            write_data(adata[data_split[typ]], f'{dirpath}/{typ}.h5ad')
        else:
            os.makedirs(f'{dirpath}/{typ}/', exist_ok=True)
            chunksize_ = len(data_split[typ])-1 if chunksize >= len(data_split[typ]) else chunksize
            for i, (start) in enumerate(range(0, len(data_split[typ]), chunksize_)):
                adata = read_data(datapath)
                adata = adata[data_split[typ][start:start+chunksize_]]
                if not isinstance(adata, AnnData):
                    adata=adata.to_adata()
                write_data(adata, f'{dirpath}/{typ}/{i}.h5ad')














        