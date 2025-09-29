setwd("/data/wuqinhua/phase/covid19/datasets")
library(Seurat)
library(SeuratDisk)

sbj <- list()
 
data_folder <- "./pre_data/17_Wilk_2021/data"
file_list <- list.files(data_folder, pattern=".rds", full.names=FALSE)

for (file_name in file_list) {
    sample_id <- substr(file_name, 1, 10) 
    sample_data <- readRDS(file.path(data_folder, file_name))
  
    sce <- CreateSeuratObject(counts = sample_data$exon, project = sample_id)
    rownames(sce@meta.data) <- paste0(rownames(sce@meta.data), "-", sample_id)
    sbj[[sample_id]] <- sce
    
}
merged_seurat <- merge(x = sbj[[1]], y = sbj[c(2:length(sbj))])

summary(merged_seurat)
# saveRDS(merged_seurat,"./pre_data/17_Wilk_2021/mobj.rds")


SaveH5Seurat(merged_seurat, file = "./pre_data/17_Wilk_2021/sce.h5Seurat")
Convert("./pre_data/17_Wilk_2021/sce.h5Seurat", dest = "h5ad")