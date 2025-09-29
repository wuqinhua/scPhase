import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData

os.chdir("/data/wuqinhua/phase/covid19/datasets")

folder_path = './pre_data/4_Combes_2021/data' 
samples = [dir_name for dir_name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, dir_name))]
print(samples)

all_adatas = []
for sample in samples:
    print(sample)
    adata = sc.read_10x_mtx('./pre_data/4_Combes_2021/data/'+sample+'',
        var_names = 'gene_symbols',
        cache=False
    )
    new_indices = [index + "-Combes2021" for index in adata.obs.index]
    adata.obs.index = new_indices
    adata.obs["batch"] = "4"
    adata.obs["sample_id"] = sample
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    all_adatas.append(adata)
adatas = sc.concat(all_adatas, axis = 0)
print(adatas)

csv_data = pd.read_csv('./pre_data/4_Combes_2021/dict_Combes.csv')

sample_ids_to_keep = csv_data['GSE'].tolist()
adatas = adatas[adatas.obs['sample_id'].isin(sample_ids_to_keep)].copy()

sampleid_group_dict1 = dict(zip(csv_data['GSE'], csv_data['Sample_ID']))
sampleid_group_dict2 = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['sample_id'] = adatas.obs['sample_id'].map(sampleid_group_dict1)

adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict2)


adatas.obs_names_make_unique

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
print(adatas)
adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]
print(adatas)

adatas.obs_names_make_unique()
results_file = './pro_data/4_Combes_2021.h5ad'
adatas.write(results_file, compression='gzip')