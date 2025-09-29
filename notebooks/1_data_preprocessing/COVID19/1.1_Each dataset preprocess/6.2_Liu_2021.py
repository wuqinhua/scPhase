import os
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/6_Liu_2021/sce.h5ad')
adata = ad.AnnData(X=adata.raw.X, var=adata.raw.var, obs=adata.obs) 
submatrix_block = adata.X[:15, :25].toarray()
submatrix_block

csv_data = pd.read_csv('./pre_data/6_Liu_2021/dict_Liu.csv')
sample_ids_to_keep = csv_data['Sample_ID'].tolist()
adatas = adata[adata.obs['sample_name'].isin(sample_ids_to_keep)].copy()

sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sample_name'].map(sampleid_group_dict)
adatas.obs

new_indices = [index + "-Liu2021" for index in adatas.obs.index]
adatas.obs.index = new_indices
adatas.obs.rename(columns={'sample_name': 'sample_id'}, inplace=True)
adatas.obs.columns

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
print(adatas)

adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]
print(adatas)

results_file = './pro_data/6_Liu_2021.h5ad'
adatas.write(results_file, compression='gzip')