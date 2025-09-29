import os
import sys
import scipy
import numpy as np
import pandas as pd
import scanpy as sc

os.chdir("/data/wuqinhua/phase/covid19")

adata = sc.read('./Alldata_hvg.h5ad')


adata.raw = adata
adata = adata[:,adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)

sc.pp.harmony_integrate(adata, 'sample_id')

adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)

results_file = './Alldata_Harmony_umap.h5ad' 
adata.write(results_file, compression='gzip')