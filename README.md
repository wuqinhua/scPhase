# scPhase: Exploring phenotype-related single-cells through attention-enhanced representation learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

**scPhase** is a deep learning framework designed to predict clinical phenotypes from single-cell RNA-sequencing (scRNA-seq) data. It treats each patient sample as a "bag" of single cells and uses a Multiple Instance Learning (MIL) approach to learn a comprehensive representation of their gene expression profiles.

The model incorporates several key architectural components to enhance performance and generalizability across diverse patient cohorts, including:
* An **Instance Encoder** with a multi-layer perceptron to learn rich gene representations.
* A **LinFormer Attention** module to efficiently capture inter-cellular relationships.
* A **Mixture-of-Experts (MoE) based MIL Aggregation** layer for robustly weighting cell importance.
* A **Domain Adaptation** module with a gradient reversal layer to mitigate batch effects.

Furthermore, scPhase includes a built-in interpretability framework that uses cellular attention and gene attribution scores (via Integrated Gradients) to identify the key cells and genes driving the phenotype predictions.

The manuscript has been pre-printed in bioRxiv:
> Qinhua Wu, Junxiang Ding, Ruikun He, Lijian Hui, Junwei Liu, Yixue Li. Exploring phenotype-related single-cells through attention-enhanced representation learning. *bioRxiv* (2024). [https://doi.org/10.1101/2024.10.31.619327](https://doi.org/10.1101/2024.10.31.619327)

#

<img src="https://github.com/wuqinhua/scPhase/blob/main/notebooks/The%20scPhase%20framwork.png" alt="架构图" width="1000"/>

***

## The scPhase pipeline

1. **Predict clinical phenotypes from scRNA-seq data**
   - 1.1 Data preprocessing: Encode the data into a format that can be read by PHASE.
   - 1.2 Gene feature embedding: Extract and represent gene features.
   - 1.3 Self-attention (SA): Learn cell embeddings.
   - 1.4 Attention-based deep multiple instance learning (AMIL): aggregate all single-cell information within a sample.
   
2. **Provide interpretability of key phenotype-related features**
   - 2.1 Attribution analysis: Use Integrated Gradients (IG) to link genes to phenotypes via attribution scores.
   - 2.2 Attention analysis: Use AMIL attention scores to relate individual cells to the phenotype.
   - 2.3 Conjoint analysis: Correlate top genes' expression levels with cells' attention scores to reveal gene-cell contributions to the phenotype.

***

## Installation
### Installing PHASE package
PHASE is written in Python and can be installed using `pip`:

```bash
pip install phase-sc
```
### Requirements
PHASE should run on any environmnet where Python is available，utilizing PyTorch for its computational needs. The training of PHASE can be done using CPUs only or GPU acceleration. If you do not have powerful GPUs available, it is possible to run using only CPUs. Before using **PHASE**, make sure the following packages are installed:

```bash
scanpy>=1.10.2  
anndata>=0.10.8  
torch>=2.4.0  
tqdm>=4.66.4  
numpy>=1.23.5  
pandas>=1.5.3  
scipy>=1.11.4  
seaborn>=0.13.2  
matplotlib>=3.6.3  
captum>=0.7.0  
scikit-learn>=1.5.1  
```
To install these dependencies, you can run the following command using `pip`:
```bash
pip install scanpy>=1.10.2 anndata>=0.10.8 torch>=2.4.0 tqdm>=4.66.4 numpy>=1.23.5 pandas>=1.5.3 scipy>=1.11.4 seaborn>=0.13.2 matplotlib==3.6.3 captum==0.7.0 scikit-learn>=1.5.1
```

Alternatively, if you are using a requirements.txt file, you can add these lines to your file and install using:
```bash
pip install -r requirements.txt
```
***

## Dataset Requirements

PHASE requires single-cell expression data to be provided as an `anndata` object in the `h5ad` format. Before initiating the training process, the dataset must undergo preprocessing, which includes the following steps:

1. **Extraction of highly variable genes**: Identify and retain the genes（default：5000） with the highest variability across cells.
2. **UMAP computation**: Generate a two-dimensional representation of the data for visualization and clustering.
3. **Cell annotation**: Annotate cell types based on the dataset, either through automated methods or manual annotation, as appropriate.

To ensure compatibility, **consistent naming** must be used within the `anndata` object for the following key pieces of information:
- **Sample IDs**: Store in `adata.obs["sample_id"]`.
- **Phenotypes**: Store in `adata.obs["phenotype"]`.
- **Cell types**: Store in `adata.obs["celltype"]`.

For detailed instructions on preprocessing and standardizing the dataset, as well as examples, please refer to the provided notebook: [preprocess_demo.ipynb](https://github.com/wuqinhua/PHASE/blob/main/Desktop/PHASE/Preprocess_demo.ipynb).

***

## Usages

### Command Line Arguments

The following table lists the command line arguments available for training the model:

| Abbreviation | Parameter      | Description                                                       |
|--------------|----------------|-------------------------------------------------------------------|
| -t           | --type         | Type of task: classification or regression.                       |
| -p           | --path         | Path to the dataset.                                              |
| -r           | --result       | Path to the directory where results will be saved.                |
| -e           | --epoch        | Number of training epochs (default: 100).                         |
| -l           | --learningrate | Learning rate for the optimizer (default: 0.00001).               |
| -d           | --devices      | List of GPU device IDs to use for training (default: first GPU).  |

Each argument is required unless a default value is specified.

### Example
```bash
PHASEtrain -t classification -p /home/user/PHASE/demo_covid.h5ad -r /home/user/PHASE/result -e 100 -l 0.00001 -d 2
```
***

## Reproduction

- **Data Preprocessing**: The folder contains preprocessing scripts and notebooks, including single-cell integration and annotation, available for both [COVID-19](https://github.com/wuqinhua/PHASE/tree/main/COVID19/1_Data_preprocess) and [Age](https://github.com/wuqinhua/PHASE/tree/main/Age/1_Data_preprocess) datasets.

- **Model Training**: Details of model training can be found in the [COVID-19 Model Training](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.1.2_Model_Training.py) and [Age Model Training](https://github.com/wuqinhua/PHASE/blob/main/Age/2_Model/2.3_Model_training.py) scripts.

- **Attribution Analysis**: 
  - For COVID-19 data, gene attribution scores can be computed using [COVID-19 Attribution Group PHASE](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.2.1_attribution_group_PHASE.py) and [COVID-19 Attribution Sample PHASE](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.2.2_attribution_sample_PHASE.py). Visualization of results is available in the [COVID-19 Attribution Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/COVID19/3_Analysis/3.1_Attribution_analysis.ipynb).
  - For Age data, gene attribution scores can be computed using [Age Attribution PHASE](https://github.com/wuqinhua/PHASE/blob/main/Age/2_Model/2.4_Attribution_PHASE.py), and result visualization is available in the [Age Attribution Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/Age/3_Analysis/3.2_Attention_analysis.ipynb).

- **Attention Analysis**: 
  - For COVID-19 data, cell attention scores can be computed using [COVID-19 Attention PHASE](https://github.com/wuqinhua/PHASE/blob/main/COVID19/2_Model/2.3_attention_cell_PHASE.py), and result visualization is available in the [COVID-19 Attention Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/COVID19/3_Analysis/3.2_Attention_analysis.ipynb).
  - For Age data, cell attention scores can be computed using [Age Attention PHASE](https://github.com/wuqinhua/PHASE/blob/main/Age/2_Model/2.5_Attention_PHASE.py), and result visualization is available in the [Age Attention Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/Age/3_Analysis/3.2_Attention_analysis.ipynb).

- **Conjoint Analysis**: Details are available in the [COVID-19 Conjoint Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/COVID19/3_Analysis/3.3_Conjoint_analysis.ipynb) and the [Age Conjoint Analysis Notebook](https://github.com/wuqinhua/PHASE/blob/main/Age/3_Analysis/3.3_Conjoint_analysis.ipynb).

***

This repository will be continuously updated during the submission process.
