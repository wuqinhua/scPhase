setwd("/data/wuqinhua/phase/covid19/datasets")
library(Seurat)
library(SeuratDisk)

sce <- readRDS('./pre_data/20_Zhu_2020/Final_nCoV_0716_upload.RDS')

sce <- UpdateSeuratObject(object =sce)
DefaultAssay(sce) <- "RNA"
rna_counts =sce@assays$RNA@counts 

colnames(sce@meta.data)[colnames(sce@meta.data) == 'batch'] <- 'sample_id'
sce@meta.data <- as.data.frame(sce@meta.data)
sce@meta.data[] <- lapply(sce@meta.data, as.character)

SaveH5Seurat(sce, file = "./pre_data/20_Zhu_2020/sce.h5Seurat", overwrite = T)
Convert("./pre_data/20_Zhu_2020/sce.h5Seurat", assay = 'RNA',dest = "h5ad", overwrite = T)