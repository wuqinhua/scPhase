import os
import scanpy as sc
import pandas as pd
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/20_Zhu_2020/sce.h5ad')
submatrix_block = adata.raw.X[:15, :25].toarray()
adata.X = adata.raw.X

csv_data = pd.read_csv('./pre_data/20_Zhu_2020/dict_Zhu.csv')
sample_ids_to_keep = csv_data['Sample_ID'].tolist()
adatas = adata[adata.obs['sample_id'].isin(sample_ids_to_keep)].copy()

sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict)

new_indices = [index + "-Zhu2020" for index in adatas.obs.index]
adatas.obs.index = new_indices
adatas.obs["batch"] = "20"

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)


adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]
print(adatas.obs)

adatas.var.rename(columns={'_index':'index'},inplace=True)
adatas.raw.var.rename(columns={'_index':'index'},inplace=True)
adatas.obsm = {}
results_file = './pro_data/20_Zhu_2020.h5ad'
adatas.write(results_file, compression='gzip')