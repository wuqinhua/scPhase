import os
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import pandas as pd
from scipy.io import mmread
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

os.chdir("/data/wuqinhua/phase/age/")

data = sc.read_h5ad("./all_pbmcs/all_pbmcs_rna.h5ad")
meta = pd.read_csv('./all_pbmcs/all_pbmcs_metadata.csv', index_col=0) 
data.obs = data.obs.join(meta)

matrix_hto = (mmread('./all_pbmcs/all_pbmcs_hto/matrix_hto.mtx'))
matrix_hto = matrix_hto.todense()
genes_hto = pd.read_csv('./all_pbmcs/all_pbmcs_hto/genes_hto.tsv', sep='\t', header=None)
barcodes_hto = pd.read_csv('./all_pbmcs/all_pbmcs_hto/barcodes_hto.tsv', sep='\t', header=None)
data_hto = pd.DataFrame(matrix_hto.T, columns=genes_hto[0], index = barcodes_hto[0])
data.obs = data.obs.join(data_hto)

sc.pp.highly_variable_genes(data, n_top_genes=5000, flavor="cell_ranger", min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'Batch')
adata = data[:, data.var.highly_variable].copy()
adata.raw = data
adata_s = adata

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver = 'arpack')
sce.pp.harmony_integrate(adata, 'Batch')
adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
sc.pp.neighbors(adata, n_pcs = 30, n_neighbors = 15)
sc.tl.umap(adata)

# adata.write('/data/wuqinhua/phase/age/all_pbmc_anno.h5ad')

cell1 = pd.read_csv("./b_cells/b_cells_metadata.csv",index_col = 0)
cell2 = pd.read_csv("./cd4_t_cells/cd4_metadata.csv",index_col = 0)
cell4 = pd.read_csv("./conventional_cd8_t_cells/conventional_cd8_metadata.csv",index_col = 0)
cell5 = pd.read_csv("./gd_t_cells/gd_t_cells_metadata.csv",index_col = 0)
cell6 = pd.read_csv("./mait_cells/mait_cells_metadata.csv",index_col = 0)
cell7 = pd.read_csv("./myeloid_cells/myeloid_cells_metadata.csv",index_col = 0)
cell8 = pd.read_csv("./nk_cells/nk_cells_metadata.csv",index_col = 0)
cell9 = pd.read_csv("./progenitor_cells/progenitor_cells_metadata.csv",index_col = 0)
cell1["Cluster_names"] = "B_" + cell1["Cluster_names"].astype(str)
cell2["Cluster_names"] = "CD4_" + cell2["Cluster_names"].astype(str)
cell4["Cluster_names"] = "CD8_" + cell4["Cluster_names"].astype(str)
cell5["Cluster_names"] = "GD_" + cell5["Cluster_names"].astype(str)
cell6["Cluster_names"] = 'MAIT'
cell7["Cluster_names"] = "Myeloid_" + cell7["Cluster_names"].astype(str)
cell8["Cluster_names"] = "NK_" + cell8["Cluster_names"].astype(str)
cell9["Cluster_names"] = 'Progenitor'
cell_anno = pd.concat([cell1,cell2,cell4,cell5,cell6,cell7,cell8,cell9], axis=0)
cell_anno = cell_anno.rename(columns={'Cluster_names': 'celltype'})
adata.obs['celltype'] = adata.obs.index.map(cell_anno['celltype'])
adata.obs['celltype'] = adata.obs['celltype'].fillna('dn_T')
adata.obs['celltype'] = adata.obs['celltype'].str.replace("/", "_")

selected_columns = adata.obs.iloc[:, list(range(19)) + [-1]]
selected_columns.to_csv("./Info/age_metadata.csv")

columns = ['Donor_id', 'Age_group', 'Sex', 'Age', 'Tube_id', 'Batch', 'File_name']
subset = adata.obs[columns]
unique_subset = subset.drop_duplicates(subset='Tube_id')
unique_subset.reset_index(drop=True, inplace=True)
unique_subset.to_csv("./Info/sample_info.csv")


gene_list = list(adata.var.index)
hvg_genes =pd.DataFrame(gene_list,columns=['genes'])
hvg_genes.to_csv("./Info/Age_hvg_genes.csv")

adata.X = adata_s.X
adata.write('./all_pbmc_anno_s.h5ad')