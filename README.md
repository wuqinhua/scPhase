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

## Workflow Overview

The scPhase workflow follows four main steps:

1.  **Data Preparation**: Format your scRNA-seq data into the required `.h5ad` file.
2.  **Configuration**: Edit the `config.json` file to specify paths and model parameters.
3.  **Execution**: Run cross-validation and interpretation analysis from the command line.
4.  **Result Interpretation**: A range of downstream biological analyses, including gene attribution, pathway enrichment, cell attention analysis, and conjoint analysis to link cellular and molecular drivers of phenotypes.
   
***

## Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/wuqinhua/scPhase.git
cd scPhase
```

Then, install the required Python packages from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

***

## Dataset Requirements

scPhase requires input data to be a single **`.h5ad`** file, which is a standard format for `scanpy` and `anndata`. The `AnnData` object within this file must be preprocessed and contain the following columns in its `.obs` attribute:

1. **Extraction of highly variable genes**: Identify and retain the genes（default：5000） with the highest variability across cells.
2. **UMAP computation**: Generate a two-dimensional representation of the data for visualization and clustering.
3. **Cell annotation**: Annotate cell types based on the dataset, either through automated methods or manual annotation, as appropriate.

Note：
* **Sample ID**: A column identifying which sample (e.g., patient) each cell belongs to.
* **Phenotype Label**: A column containing the clinical phenotype for each sample.
* **Batch Information**: A column indicating the batch, study, or cohort for each sample, which is used for domain adaptation.

The names of these columns can be specified in the `config.json` file.

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
