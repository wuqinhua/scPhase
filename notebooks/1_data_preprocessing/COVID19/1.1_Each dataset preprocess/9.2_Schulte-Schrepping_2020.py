import os
import scanpy as sc
import pandas as pd
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19/datasets")

all_adatas = []
adata1 = ad.read_h5ad('./pre_data/9_Schulte-Schrepping_2020/sce1.h5ad')
adata2 = ad.read_h5ad('./pre_data/9_Schulte-Schrepping_2020/sce2.h5ad')

adata1 = ad.AnnData(X=adata1.raw.X, var=adata1.raw.var, obs=adata1.obs) 
adata2 = ad.AnnData(X=adata2.raw.X, var=adata2.raw.var, obs=adata2.obs) 
all_adatas.append(adata1) 
all_adatas.append(adata2)
adata = adata1.concatenate(adata2,join="outer")



new_indices = [index + "-Schulte2020" for index in adata.obs.index]
adata.obs.index = new_indices
adata.obs["batch"] = "9" 

csv_data = pd.read_csv('./pre_data/9_Schulte-Schrepping_2020/dict_Schulte.csv')
sample_ids_to_keep = csv_data['sampleID'].tolist()
adatas = adata[adata.obs['sampleID'].isin(sample_ids_to_keep)].copy()

sampleid_group_dict = dict(zip(csv_data['sampleID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sampleID'].map(sampleid_group_dict)
adatas.obs.rename(columns={'sampleID': 'sample_id'}, inplace=True)
print(adatas)

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
print(adatas)
adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adata_s = adatas[adatas.obs.pct_counts_MT<20, :]
print(adata_s)

adata_s.obsm = {}

results_file = './pro_data/9_Schulte_Schrepping_2020.h5ad'
adata_s.write(results_file, compression='gzip')                                                                                                             