import os
import sys
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
import celltypist
from celltypist import models
import seaborn as sns
import scipy.io as sio
import scanpy.external as sce
import matplotlib.pyplot as plt


os.chdir("/data/wuqinhua/phase/covid19")

adata = sc.read('./Alldata.h5ad')

selected_batches = ['0', '1', '2', '3', '4','5','6', '7', '8', '9', '10','11']
adata = adata[adata.obs['batch'].isin(selected_batches)]

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.X.expm1().sum(axis = 1)
model = models.Model.load(model = 'Healthy_COVID19_PBMC.pkl')
print("model")
predictions = celltypist.annotate(adata, model = 'Healthy_COVID19_PBMC.pkl', majority_voting = True)
print("predictions")
print(predictions.predicted_labels)
adata = predictions.to_adata()
print("predictions.to_adata")


adatas = sc.read('./Alldata_Harmony_umap.h5ad')
adatas.obs["predicted_labels"] = adata.obs["predicted_labels"]
adatas.obs["over_clustering"] = adata.obs["over_clustering"]
adatas.obs["majority_voting"] = adata.obs["majority_voting"]

adata.obs.to_csv("./metadata.csv")
adatas.write("./Alldata_COVID.h5ad")