setwd("/data/wuqinhua/phase/covid19/datasets")
library(Seurat)
library(SeuratDisk)

sce1 = readRDS("./pre_data/9_Schulte-Schrepping_2020/seurat_COVID19_PBMC_cohort1_10x_jonas_FG_2020-08-15.rds")
sce2 = readRDS("./pre_data/9_Schulte-Schrepping_2020/seurat_COVID19_PBMC_jonas_FG_2020-07-23.rds")

sce1@meta.data <- as.data.frame(sce1@meta.data)
sce1@meta.data[] <- lapply(sce1@meta.data, as.character)
colnames(sce1@meta.data)[colnames(sce1@meta.data) == 'sample_id'] <- 'sampleID'

sce2@meta.data <- as.data.frame(sce2@meta.data)
sce2@meta.data[] <- lapply(sce2@meta.data, as.character)
colnames(sce2@meta.data)[colnames(sce2@meta.data) == 'sample_id'] <- 'sampleID'


DefaultAssay(sce1) <- "RNA"
DefaultAssay(sce2) <- "RNA"
sce1@assays$RNA@data = sce1@assays$RNA@counts
sce2@assays$RNA@data = sce2@assays$RNA@counts

SaveH5Seurat(sce1, file = "./pre_data/9_Schulte-Schrepping_2020/sce1.h5Seurat", overwrite=T)
Convert("./pre_data/9_Schulte-Schrepping_2020/sce1.h5Seurat", dest = "h5ad",overwrite=T)
SaveH5Seurat(sce2, file = "./pre_data/9_Schulte-Schrepping_2020/sce2.h5Seurat", overwrite=T)
Convert("./pre_data/9_Schulte-Schrepping_2020/sce2.h5Seurat", dest = "h5ad",overwrite=T)