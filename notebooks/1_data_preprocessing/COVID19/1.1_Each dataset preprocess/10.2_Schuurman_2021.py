import os
import scanpy as sc
import pandas as pd
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/10_Schuurman_2021/pbmc.h5ad')

adata.obs.rename(columns={'id_new': 'sample_id'}, inplace=True)
adata.obs = adata.obs.rename(index=lambda x: x.replace('.', '-'))
csv_data = pd.read_csv('./pre_data/10_Schuurman_2021/dict_Schuurman.csv')
sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adata.obs['group'] = adata.obs['sample_id'].map(sampleid_group_dict)

adata.obs['batch'] = 10
adata.obs["batch"]=adata.obs["batch"].astype("category") 

new_indices = [index + "-Schuurman2021" for index in adata.obs.index]
adata.obs.index = new_indices

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adatas.var.rename(columns={'_index':'index'},inplace=True)
adatas.raw.var.rename(columns={'_index':'index'},inplace=True)

results_file = './pro_data/10_Schuurman_2021.h5ad'
adatas.write(results_file, compression='gzip')