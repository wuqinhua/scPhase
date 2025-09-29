import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import os

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/19_Zhao_2021/seu_obj.h5ad')
adata

adata.obs["Sample"] =  adata.obs['patient'].astype(str) + '_' + adata.obs['Sample'].astype(str)
adata.obs.rename(columns={'Sample': 'sample_id'}, inplace=True)
adata.obs_names_make_unique()

adata.obs.index = adata.obs.index +"-Zhao2021"
adata.obs["batch"] = "19"
adata.obs = adata.obs.astype(str)                                                                                                       .astype(str)

csv_data = pd.read_csv('./pre_data/19_Zhao_2021/dict_Zhao.csv')
sample_ids_to_keep = csv_data['Sample_ID'].tolist()
adatas = adata[adata.obs['sample_id'].isin(sample_ids_to_keep)].copy()



sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict)
print(adatas)


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


results_file = './pro_data/19_Zhao_2021.h5ad'
adatas.write(results_file, compression='gzip')
