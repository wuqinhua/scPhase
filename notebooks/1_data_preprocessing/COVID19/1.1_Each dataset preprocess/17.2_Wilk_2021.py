import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import os

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/17_Wilk_2021/sce.h5ad')


csv_data = pd.read_csv('./pre_data/17_Wilk_2021/dict_wilk.csv')
sample_ids_to_keep = csv_data['GSE'].tolist()
adata.obs.rename(columns={'orig.ident': 'sample_id'}, inplace=True)
adatas = adata[adata.obs['sample_id'].isin(sample_ids_to_keep)].copy()

new_indices = [index + "-Wilk2021" for index in adatas.obs.index]
adatas.obs.index = new_indices
adatas.obs["batch"] = "17"

sampleid_group_dict1 = dict(zip(csv_data['GSE'], csv_data['Sample_ID']))
sampleid_group_dict2 = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['sample_id'] = adatas.obs['sample_id'].map(sampleid_group_dict1)
adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict2)
print(adatas)

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
print(adatas)
adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]

print(adatas)

adatas.var.rename(columns={'_index':'index'},inplace=True)
adatas.raw.var.rename(columns={'_index':'index'},inplace=True)
results_file = './pro_data/17_Wilk_2021.h5ad'
adatas.write(results_file, compression='gzip')