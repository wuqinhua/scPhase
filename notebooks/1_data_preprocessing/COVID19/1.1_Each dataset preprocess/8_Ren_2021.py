import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19/datasets")

folder_path = './pre_data/8_Ren_2021/pbmc1'
samples = [dir_name for dir_name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, dir_name))]
print(samples)


all_adatas = []
for sample in samples:
    print(sample)
    adata = sc.read_10x_mtx(
        os.path.join(folder_path, sample, 'filtered_feature_bc_matrix'),
        var_names='gene_symbols',  
        cache=True
    )
    new_indices = [index + "-Ren2021" for index in adata.obs.index]
    adata.obs.index = new_indices
    adata.obs["batch"] = "8"
    adata.obs["sample_id"] = sample
    all_adatas.append(adata)
adatas = sc.concat(all_adatas)
adatas  

csv_data2 = pd.read_csv('./pre_data/8_Ren_2021/dict_Ren_zzm.csv')
sampleid_group_dict2 = dict(zip(csv_data2['Sample_ID'], csv_data2['group']))
sample_ids_to_keep2 = csv_data2['Sample_ID'].tolist()
adatas2 = adatas[adatas.obs['sample_id'].isin(sample_ids_to_keep2)].copy()
adatas2.obs['group'] = adatas2.obs['sample_id'].map(sampleid_group_dict2)
print(adatas2)

sc.pp.filter_cells(adatas2, min_genes=200)
sc.pp.filter_genes(adatas2, min_cells=3)
print(adatas2)

adatas2.var['MT'] = adatas2.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas2, qc_vars=['MT'], percent_top=None, log1p=False, inplace=True)
adatas2 = adatas2[adatas2.obs.pct_counts_MT<10, :]
print(adatas2)



results_file2 = './pro_data/8_Ren_2021_zzm.h5ad'
adatas2.write(results_file2, compression='gzip')