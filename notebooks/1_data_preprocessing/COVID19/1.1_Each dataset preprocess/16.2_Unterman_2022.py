import os
import numpy as np
import pandas as pd
import scanpy as sc

os.chdir("/data/wuqinhua/phase/covid19/datasets")

samples = ["NS0A", "NS0B","NS1A", "NS1B","TP6A", "TP6B","TP7A", "TP7B","TP8B", "TP9B","TS2A", "TS2B","TS3A", "TS3B","TS4A", "TS4B","TS5A", "TS5B"]
all_adatas = []
for sample in samples:
    adata = sc.read_10x_mtx('./pre_data/16_Unterman_2022/data/'+sample+'_HHT_cellranger/filtered_feature_bc_matrix',
        var_names = 'gene_symbols',
        cache=True
    )
    new_indices = [index + "-Unterman2022" for index in adata.obs.index]
    adata.obs.index = new_indices
    adata.obs["batch"] = "16"
    adata.obs["sample_id"] = sample
    all_adatas.append(adata)
adatas = sc.concat(all_adatas)
adatas

csv_data = pd.read_csv('./pre_data/16_Unterman_2022/dict_Unterman.csv')
sample_ids_to_keep = csv_data['Sample_ID'].tolist()
adatas = adatas[adatas.obs['sample_id'].isin(sample_ids_to_keep)].copy()

sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict)
print(adatas)

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
print(adatas)
adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]

adatas.obs_names_make_unique()
print(adatas)

results_file = './pro_data/16_Unterman_2022.h5ad'
adatas.write(results_file, compression='gzip')

