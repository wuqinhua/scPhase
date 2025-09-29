setwd("/data/wuqinhua/phase/covid19/datasets")
library(Seurat)
library(SeuratDisk)

data <-read.csv("./pre_data/10_Schuurman_2021/GSE164948_covid_control_RNA_counts.csv.gz",row.names = 1)
pbmc = t(data)
pbmc <- CreateSeuratObject(counts = data)

rownames_pbmc <- rownames(pbmc@meta.data)
rownames_pbmc <- gsub("\\.", "-", rownames_pbmc)
rownames(pbmc@meta.data) <- rownames_pbmc

metadata <-read.csv("./pre_data/10_Schuurman_2021/GSE164948_covid_control_count_metadata.csv.gz",sep = ";",row.names = 1)
pbmc@meta.data = metadata

column_names <- colnames(pbmc@meta.data)
sample_counts <- table(pbmc@meta.data$id_new)

rna_counts <- pbmc@assays$RNA@counts


original_colnames <- colnames(rna_counts)
new_colnames <- gsub("\\.", "-", original_colnames)
colnames(rna_counts) <- new_colnames
head(rna_counts)

pbmc@assays$RNA@counts = rna_counts

SaveH5Seurat(pbmc, file = "./pre_data/10_Schuurman_2021/pbmc.h5Seurat")
Convert("./pre_data/10_Schuurman_2021/pbmc.h5Seurat", dest = "h5ad")