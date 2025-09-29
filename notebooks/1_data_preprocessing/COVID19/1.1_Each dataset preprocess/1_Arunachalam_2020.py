import os
import numpy as np
import pandas as pd
import scanpy as sc

os.chdir("/data/wuqinhua/phase/covid19/datasets")

samples = ["cov01", "cov02","cov03", "cov04","cov07", "cov08","cov09", "cov10","cov11", "cov12","cov17", "cov18"]
all_adatas = []
for sample in samples:
    adata = sc.read_10x_mtx('./pre_data/1_Arunachalam_2020/'+sample+'',
        var_names = 'gene_symbols',
        cache=True
    )
    new_indices = [index + "-Arunachalam2020" for index in adata.obs.index]
    adata.obs.index = new_indices
    adata.obs["batch"] = "4"
    adata.obs["sample_id"] = sample
    all_adatas.append(adata)
adatas = sc.concat(all_adatas)
adatas

csv_data = pd.read_csv('./1_Arunachalam_2020/dict_Arunachalam.csv')
sampleid_group_dict = dict(zip(csv_data['Sample_ID'], csv_data['group']))
adatas.obs['group'] = adatas.obs['sample_id'].map(sampleid_group_dict)
adatas.obs['group']
adatas

sc.pp.filter_cells(adatas, min_genes=200)
sc.pp.filter_genes(adatas, min_cells=3)
adatas

adatas.var['MT'] = adatas.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas = adatas[adatas.obs.pct_counts_MT<10, :]
adatas

adatas.obs_names_make_unique()
results_file = './pro_data/1_Arunachalam_2020.h5ad'
adatas.write(results_file, compression='gzip')