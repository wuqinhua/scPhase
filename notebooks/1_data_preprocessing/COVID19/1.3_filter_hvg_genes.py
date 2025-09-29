import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

os.chdir("/data/wuqinhua/phase/covid19")

adata = ad.read_h5ad("./Alldata.h5ad")

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
batch_key = "batch"
sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger", batch_key= batch_key,layer=None)

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
adata_hvg
hvg_genes =pd.DataFrame(adata_hvg.var.index)
hvg_genes.to_csv("./hvg_genes.csv")

results_file = '/Alldata_hvg.h5ad' 
adata_hvg.write(results_file, compression='gzip')