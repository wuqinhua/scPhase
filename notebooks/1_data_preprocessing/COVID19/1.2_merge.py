import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import hdf5plugin
from anndata import AnnData

os.chdir("/data/wuqinhua/phase/covid19/datasets/pro_data")

adata1 = ad.read_h5ad('./1_Arunachalam_2020.h5ad')
adata3 = ad.read_h5ad('./3_COMBAT_2022.h5ad')
adata4 = ad.read_h5ad('./4_Combes_2021.h5ad')
adata6 = ad.read_h5ad('./6_Liu_2021.h5ad')
adata8 = ad.read_h5ad('./8_Ren_2021_zzm.h5ad')
adata9 = ad.read_h5ad('./9_Schulte_Schrepping_2020.h5ad')
adata10 = ad.read_h5ad('./10_Schuurman_2021.h5ad')
adata13 = ad.read_h5ad('./13_Stephenson_2021.h5ad')
adata16 = ad.read_h5ad('./16_Unterman_2022.h5ad')
adata17 = ad.read_h5ad('./17_Wilk_2021.h5ad')
adata19 = ad.read_h5ad('./19_Zhao_2021.h5ad')
adata20 = ad.read_h5ad('./20_Zhu_2020.h5ad')
adata1.raw = adata1
adata3.raw = adata3
adata4.raw= adata4
adata6.raw = adata6
adata8.raw = adata8
adata9.raw = adata9
adata10.raw = adata10
adata13.raw = adata13
adata16.raw = adata16
adata17.raw = adata17
adata19.raw = adata19
adata20.raw = adata20
print('read_over')

merged_adata = adata1.concatenate(adata3, adata4, adata6, adata8, adata9, adata10, adata13, adata16, adata19, adata17, adata20,join='outer',batch_key='batch')
print('meage_over')

merged_adata.var = merged_adata.var.astype(str)
merged_adata.obs = merged_adata.obs.astype(str)

sc.pp.calculate_qc_metrics(merged_adata, percent_top=None, log1p=False, inplace=True)
merged_adata.obs = merged_adata.obs.iloc[:, :8]
merged_adata.var = merged_adata.var.iloc[:, 92:96]

results_file = '/data/wuqinhua/phase/covid19/Alldata.h5ad' 
merged_adata.write(results_file, compression='gzip')
print('save_over')