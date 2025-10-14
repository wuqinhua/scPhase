# data_loader.py
import os
import pickle
import torch
import numpy as np
import scanpy as sc
from tqdm import tqdm  
import logging
import scipy.sparse
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

def load_data(config):
    path_cfg = config['path_params']
    master_pickle_path = path_cfg.get('_master_pickle_file')

    if master_pickle_path and os.path.exists(master_pickle_path):
        logger.info(f"Loading preprocessed data from master pickle file: {master_pickle_path}")
        with open(master_pickle_path, 'rb') as f:
            master_data = pickle.load(f)
    else:
        if master_pickle_path:
            logger.info(f"Master pickle file not found at {master_pickle_path}. Processing from h5ad file.")
        else:
            logger.info("No master pickle file configured. Processing directly from h5ad file.")
        master_data = _create_master_pickle(config)

    DataList = master_data['data_list_dense']
    DataLabel = np.array(master_data['labels'])
    DataBatch = np.array(master_data['groups']).astype(int)
    SampleIDs = master_data.get('sample_ids', []) 
    if not SampleIDs:
        logger.warning("Sample IDs were not found in the loaded data.")

    return DataList, DataLabel, DataBatch, SampleIDs


def _create_master_pickle(config: dict):
    path_cfg = config['path_params']
    run_cfg = config['run_params']
    data_cfg = config.get('data_params', {})
    
    h5ad_path = path_cfg['data_h5ad_file']
    task_type = run_cfg.get('task_type', 'classification')
    
    sample_col = data_cfg.get('sample_col', 'sample_id')
    label_col = data_cfg.get('label_col', 'phenotype')
    batch_col = data_cfg.get('batch_col', 'batch')
    
    logger.info(f"Processing data directly from: {h5ad_path}")
    traindata = sc.read_h5ad(h5ad_path)
    
    has_batch_col = batch_col in traindata.obs.columns
    if not has_batch_col:
        logger.warning(f"Batch column '{batch_col}' not found. Assigning default batch value 0 to all samples.")
    
    label_processor = None
    if task_type == 'classification':
        unique_labels = sorted(traindata.obs[label_col].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        logger.info(f"Auto-generated label map for classification: {label_map}")
        label_processor = lambda l: label_map.get(l, -1)
    else: # Regression
        logger.info("Processing labels for regression task.")
        label_processor = float
        
    data_list_sparse = []
    sample_labels = []
    sample_groups = []
    sample_ids_list = [] 

    grouped = traindata.obs.groupby(sample_col)
    pbar = tqdm(grouped, desc="Processing samples into memory", total=len(grouped))
    
    for sample_id, sample_obs in pbar:
        sample_ids_list.append(sample_id) 
        sample_labels.append(label_processor(sample_obs[label_col].iloc[0]))
        sample_groups.append(sample_obs[batch_col].iloc[0] if has_batch_col else 0)
        
        sample_data = traindata[sample_obs.index].X
        if scipy.sparse.issparse(sample_data):
            data_list_sparse.append(sample_data.tocsr())
        else:
            data_list_sparse.append(scipy.sparse.csr_matrix(sample_data))
        
    data_list_dense = [sparse.toarray() for sparse in data_list_sparse]
        
    master_data = {
        'data_list_dense': data_list_dense, 
        'labels': np.array(sample_labels),
        'groups': np.array(sample_groups),
        'sample_ids': sample_ids_list, 
    }

    master_pickle_path = path_cfg.get('_master_pickle_file')
    if master_pickle_path:
        logger.info(f"Saving master data object to: {master_pickle_path}")
        os.makedirs(os.path.dirname(master_pickle_path), exist_ok=True)
        with open(master_pickle_path, 'wb') as f:
            pickle.dump(master_data, f)

    logger.info(f"Data processing complete. Processed {len(data_list_dense)} samples.")
    return master_data


def sparse_collate_fn(batch):

    data_list, label_list, batch_list, sample_ids_list = [], [], [], []

    for data, label, batch_label, sample_id in batch:
        if hasattr(data, 'is_sparse') and data.is_sparse:
            data = data.to_dense()
        data_list.append(data)
        label_list.append(label)
        batch_list.append(batch_label)
        sample_ids_list.append(sample_id)
    
    batched_data = torch.stack(data_list)
    batched_labels = torch.as_tensor(label_list)
    batched_batch_labels = torch.as_tensor(batch_list)
    
    return batched_data, batched_labels, batched_batch_labels, sample_ids_list

class AttnMoE_Dataset(Dataset):
    def __init__(self, DataList, DataLabel, DataBatch, SampleIDs):
        super().__init__()
        self.DataList = DataList
        self.DataLabel = DataLabel
        self.DataBatch = DataBatch
        self.SampleIDs = SampleIDs 
        self.aligned_data_cache = {}
        
    def align_datalist(self, data, index):
        if index in self.aligned_data_cache:
            return self.aligned_data_cache[index]
            
        X = csr_matrix(data)
        sparse_tensor = torch.sparse_coo_tensor(
            indices=torch.from_numpy(np.vstack([X.tocoo().row, X.tocoo().col])),
            values=torch.from_numpy(X.tocoo().data),
            size=X.shape
        ).float()
        
        self.aligned_data_cache[index] = sparse_tensor
        return sparse_tensor
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        data = self.DataList[index]
        dataList = self.align_datalist(data, index)
        dataLabel = self.DataLabel[index]
        dataBatch = self.DataBatch[index]
        sampleID = self.SampleIDs[index] 

        return dataList, dataLabel, dataBatch, sampleID

    def __len__(self):
        return len(self.DataLabel)
