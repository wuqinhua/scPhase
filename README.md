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

The manuscript has been published in Genome Medicine:
> Qinhua Wu, Junxiang Ding, Ruikun He, Lijian Hui, Junwei Liu, Yixue Li. Exploring phenotype-related single-cells through attention-enhanced representation learning. *Genome Medicine* (2026). [https://link.springer.com/article/10.1186/s13073-026-01598-x](https://link.springer.com/article/10.1186/s13073-026-01598-x)

#

<img src="https://github.com/wuqinhua/scPhase/blob/main/Figure1.png" alt="架构图" width="1200"/>

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

scPhase requires your single-cell data to be provided as a single, preprocessed **`.h5ad` file**. The `AnnData` object within this file must contain specific metadata in its `.obs` attribute for the model to function correctly.

#### Required `AnnData` Structure

The `.obs` DataFrame of your `AnnData` object must include the following columns:

* **Sample ID**: A column that identifies which sample or patient each cell belongs to. The default expected name is `sample_id`.
* **Phenotype**: A column containing the clinical outcome for each sample. This can be a categorical label for classification or a continuous value for regression. The default expected name is `phenotype`.
* **Batch**: A column indicating the batch, study, or cohort for each sample. This is used by the domain adaptation module to correct for technical variations. The default expected name is `batch`.

You can easily customize these default column names in the `data_params` section of your `config.json` file.

#### Recommended Preprocessing Workflow

To prepare your data into the required format, we recommend the following workflow:

1.  **Feature Selection**: The model expects a fixed number of input features. First, identify a set of highly variable genes (HVGs) from your dataset. The default input dimension for the model is **5000 HVGs**. Your final `adata.X` matrix should be subsetted to contain only these genes.
2.  **Cell Type Annotation**: Annotating your cells with their respective types (e.g., T-cells, B-cells) is crucial for the downstream interpretation of the model's results. These annotations should be stored in an `.obs` column.
3.  **UMAP Computation**: While not used for training, computing a UMAP embedding is necessary for visualizing the cell attention scores generated during the interpretation phase. The result should be stored in `adata.obsm['X_umap']`.


***


## Data Availability

All processed data, including trained models and interpretability metrics, have been deposited in Zenodo and can be accessed at: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18059900.svg)](https://doi.org/10.5281/zenodo.18059900)

## Usages

The entire workflow is controlled by the `run.py` script and a central `config.json` file.

### Step 1: Configure Your Experiment

Before running, edit `config.json` to match your data and experimental setup. The most important parameters to change are:

| Parameter               | Section         | Description                                                               |
| ----------------------- | --------------- | ------------------------------------------------------------------------- |
| `data_h5ad_file`        | `path_params`   | **Required**: The absolute path to your input `.h5ad` data file.      |
| `RESULTS_DIR`           | `path_params`   | **Required**: Path to the directory where all output files will be saved.      |
| `MODEL_NAME`            | `path_params`   | A name for your experiment (e.g., "COVID19", "Aging").                    |
| `sample_col`            | `data_params`   | The column name in `.obs` for sample IDs.                             |
| `label_col`             | `data_params`   | The column name in `.obs` for phenotype labels.                         |
| `batch_col`             | `data_params`   | The column name in `.obs` for batch/cohort information.                |
| `device_model`          | `run_params`    | The primary GPU for the model (e.g., "cuda:0").                           |
| `device_encoder`        | `run_params`    | A secondary GPU for the attention encoder (can be the same as `device_model`). |
| `task_type`             | `run_params`    | The prediction task type: `classification` or `regression`.               |
| `n_classes`             | `model_params`  |Number of prediction output classes. E.g., 3 for a 3-class classification task, or 1 for regression. |

### Step 2: Run the Workflow

Use the command line to execute either the cross-validation (training) or the interpretation analysis.

#### To Run Cross-Validation Training:
This command trains the model using the cross-validation strategy defined in `run_cv.py`. It uses Leave-One-Group-Out if multiple batches are detected; otherwise, it uses k-fold CV.

```bash
python run.py --config config.json --cv
```

#### To Run Interpretation Analysis:
This command must be run **after** training is complete. It loads the saved models from each fold and computes ensemble interpretability results.

```bash
python run.py --config config.json --interpret
```

***

## Interpreting the Output

All results are saved in the directory specified by `RESULTS_DIR`[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18059900.svg)](https://doi.org/10.5281/zenodo.18059900). Key outputs include:

* **Trained Models**: The best model from each CV fold is saved as `BestModel_{MODEL_NAME}_Fold{N}.pt`.
* **Performance Metrics**:
    * `AllFolds_{MODEL_NAME}.csv`: Detailed performance metrics for each validation fold.
    * `Summary_{MODEL_NAME}.csv`: Aggregated performance (mean ± std) across all folds.
* **Interpretability Results**:
    * `ensemble_gene_attributions_{phenotype}.csv` / `.pdf`: Tables and bar plots of the top-ranked genes associated with each phenotype.
    * `sample_gene_attribution_mean.csv`: A sample-by-gene matrix of mean attribution scores across folds.
    * `ensemble_adata_with_attention.h5ad`: An `AnnData` object with the mean cell attention scores from all folds saved in `.obs['attention_weight_mean']`.
    * `UMAP Plots (*.pdf)`: UMAP visualizations colored by cell type and cell attention, revealing which cells the model focused on for its predictions.


***

## Citation

If you use scPhase in your research, please cite our manuscript:

> Qinhua Wu, Junxiang Ding, Ruikun He, Lijian Hui, Junwei Liu, Yixue Li. Exploring phenotype-related single-cells through attention-enhanced representation learning. *bioRxiv* (2024). https://doi.org/10.1101/2024.10.31.619327

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
