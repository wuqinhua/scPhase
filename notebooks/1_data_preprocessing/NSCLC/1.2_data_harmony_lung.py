import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
adata = sc.read_h5ad("/data/wuqinhua/phase/LungCancerCell/Lung_CancerCell.h5ad")
logging.info("read_over")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
logging.info("pca_over")

sc.external.pp.harmony_integrate(adata,"sample_id")
logging.info("harmony_over")

adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)
logging.info("umap_over")

adata.write_h5ad("/data/wuqinhua/phase/LungCancerCell/Lung_CancerCell_harmony.h5ad", compression="gzip")
logging.info("save_over")