import os
import scanpy as sc
import pandas as pd
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19/datasets")

adata = ad.read_h5ad('./pre_data/3_COMBAT_2022/COMBAT-CITESeq-EXPRESSION-ATLAS.h5ad')
# adata

csv_data = pd.read_csv('./pre_data/3_COMBAT_2022/dict_combat.csv')
sample_ids_to_keep = csv_data['Sample_ID'].tolist()
adatas = adata[adata.obs['scRNASeq_sample_ID'].str[:12].isin(sample_ids_to_keep)].copy()
# print(adatas)

sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['scRNASeq_sample_ID'].str[:12].map(sampleid_group_dict)

new_indices = [index + "-COMBAT2022" for index in adatas.obs.index]
adatas.obs.index = new_indices
adatas.obs["batch"] = "3"
adatas.obs.rename(columns={"scRNASeq_sample_ID": "sample_id"}, inplace=True)
print(adatas)

adatas.X = adatas.layers["raw"]

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
sc.pp.filter_cells(adata, min_counts = 0)
print(adatas)

adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]
print(adatas)

adatas.uns = {}
adatas.layers = {}
adatas.obsm = {}


results_file = './pro_data/3_COMBAT_2022.h5ad'
adatas.write(results_file, compression='gzip')