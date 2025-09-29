import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/13_Stephenson_2021/haniffa21_Stephenson_2021.h5ad')
adata

csv_data = pd.read_csv('./pre_data/13_Stephenson_2021/dict_Stephenson.csv')
sample_ids_to_keep = csv_data['Sample_ID'].tolist()
adatas = adata[adata.obs['sample_id'].isin(sample_ids_to_keep)].copy()
print(adatas)

sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict)
    
new_indices = [index + "-Stephenson2021" for index in adatas.obs.index]
adatas.obs.index = new_indices
adatas.obs["batch"] = "13"
print(adatas)

adatas.X = adatas.layers["raw"]

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
print(adatas)

adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]

adatas.uns = {}
adatas.layers = {}
adatas.obsm = {}
print(adatas)

results_file = './pro_data/13_Stephenson_2021.h5ad'
adatas.write(results_file, compression='gzip')