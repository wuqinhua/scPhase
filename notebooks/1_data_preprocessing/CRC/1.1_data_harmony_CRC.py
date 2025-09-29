import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import logging

adata = sc.read_10x_h5("/data/wuqinhua/phase/CRC/CRC_Cell_raw.h5")
logging.info("read_over")

ct = pd.read_csv('/data/wuqinhua/phase/CRC/GSE178341_crc10x_full_c295v4_submit_cluster.csv',index_col=0)
mt = pd.read_csv('/data/wuqinhua/phase/CRC/GSE178341_crc10x_full_c295v4_submit_metatables.csv',index_col=0)

merged_meta = pd.merge(mt, ct, left_index=True, right_index=True, how='outer')
adata.obs = pd.merge(adata.obs,merged_meta, left_index=True, right_index=True, how='outer')

adata.obs = adata.obs.rename(columns={"PatientTypeID": "sample_id", "MMRStatus": "phenotype", "clMidwayPr": "celltype"})
adata.var_names_make_unique() 

cat_column = adata.obs['phenotype']
if pd.api.types.is_categorical_dtype(cat_column) and 'normal' not in cat_column.cat.categories:
    adata.obs['phenotype'] = cat_column.cat.add_categories('normal')
adata.obs['phenotype'] = adata.obs['phenotype'].fillna('normal')

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger", min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample_id')
adata.raw = adata
adata_s = adata[:, adata.var.highly_variable].copy()
logging.info("hvg_over")


sc.pp.scale(adata_s, max_value=10)
sc.tl.pca(adata_s, svd_solver='arpack')
sc.pp.neighbors(adata_s, n_neighbors=10, n_pcs=50)

sc.external.pp.harmony_integrate(adata_s, 'sample_id')
logging.info("harmony_over")

adata_s.obsm['X_pca'] = adata_s.obsm['X_pca_harmony']
sc.pp.neighbors(adata_s, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata_s)
logging.info("umap_over")

results_file = '/data/wuqinhua/phase/CRC/CRC_MMR.h5ad' 
adata_s.write(results_file, compression='gzip')
logging.info("save_over")
