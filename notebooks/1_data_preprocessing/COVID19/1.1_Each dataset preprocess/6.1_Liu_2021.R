setwd("/data/wuqinhua/phase/covid19/datasets")

library(Seurat)
library(SeuratDisk)

pbmc <- readRDS('./pre_data/6_Liu_2021/GSE161918_AllBatches_SeuratObj.rds')
sce <- UpdateSeuratObject(object = pbmc)

DefaultAssay(sce)
DefaultAssay(sce) <- "RNA"
DefaultAssay(sce)

sce@meta.data <- as.data.frame(sce@meta.data)
sce@meta.data[] <- lapply(sce@meta.data, as.character)
colnames(sce@meta.data)[colnames(sce@meta.data) == 'sample_id'] <- 'sample_name1'

SaveH5Seurat(sce, file = "./pre_data/6_Liu_2021/sce.h5Seurat", overwrite=T)
Convert("./pre_data/6_Liu_2021/sce.h5Seurat", dest = "h5ad",overwrite=T)