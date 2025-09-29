# run_all.py
import os
import json
import logging
import argparse
import glob
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from torch.utils.data import DataLoader
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from scphase.data_loader import load_data, AttnMoE_Dataset, sparse_collate_fn
from scphase.run_cv import run_cv_experiment
from scphase.model import SCMIL_AttnMoE
from scphase.train_utils import (
    setup_logging,
    set_seed,
    calculate_gene_attributions,
    calculate_cell_attention,
    ensemble_gene_attributions,
    ensemble_cell_attentions,
    plot_ensemble_cell_attention_umaps,
    plot_ensemble_gene_attributions,
    calculate_sample_gene_attributions,
    ensemble_sample_gene_attributions
)

def run_interpretation_ensemble(config_path):

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger = logging.getLogger("SCMIL_Workflow")
    logger.info("--- Starting Ensemble Interpretability Analysis ---")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Forcing CPU.")
        config["run_params"]["device_model"] = "cpu"
        config["run_params"]["device_encoder"] = "cpu"
    
    path_cfg = config['path_params']
    run_cfg = config['run_params']
    model_cfg = config['model_params']
    device = run_cfg['device_model']

    pattern = os.path.join(path_cfg['RESULTS_DIR'], f"BestModel_{path_cfg['MODEL_NAME']}_Fold*.pt")
    fold_model_paths = sorted(glob.glob(pattern), key=lambda x: int(x.split('Fold')[1].split('.')[0]))
    
    if not fold_model_paths:
        logger.warning(f"No models found at {pattern}. Running cross-validation first.")
        logger.info("--- Auto Action: Starting Cross-Validation ---")
        run_cv_experiment(config_path)
   
        fold_model_paths = sorted(glob.glob(pattern), key=lambda x: int(x.split('Fold')[1].split('.')[0]))
        if not fold_model_paths:
            raise FileNotFoundError(f"Cross-validation completed but still no models found at {pattern}.")
    
    logger.info(f"Found {len(fold_model_paths)} fold models for ensemble analysis.")

    DataList, DataLabel, DataBatch, sample_ids = load_data(config)
    adata = sc.read_h5ad(config['path_params']['data_h5ad_file'])  
    gene_list = adata.var_names.tolist()
    
    label_map = None
    if run_cfg['task_type'] == 'classification':
        data_cfg = config.get('data_params', {})
        label_col = data_cfg.get('label_col', 'phenotype')
        
        unique_phenotypes = sorted(adata.obs[label_col].unique())
        label_map = {i: phenotype for i, phenotype in enumerate(unique_phenotypes)}
        logger.info(f"Created label map with real phenotype names: {label_map}")

    all_fold_attributions = []
    all_fold_attention_arrays = []  
    all_fold_sample_attributions = []
 
    for fold_idx, model_path in enumerate(fold_model_paths):
        logger.info(f"Processing fold {fold_idx + 1}/{len(fold_model_paths)}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            domain_classifier_weight_key = 'domain_classifier.domain_classifier.6.weight'
            if domain_classifier_weight_key in checkpoint:
                num_domains = checkpoint[domain_classifier_weight_key].shape[0]
                logger.info(f"Inferred num_domains={num_domains} from saved model for fold {fold_idx + 1}")
            else:
                num_domains = len(np.unique(DataBatch))
                logger.warning(f"Could not infer num_domains from model, using {num_domains} from data")
        except Exception as e:
            logger.error(f"Error loading checkpoint to infer num_domains: {e}")
            num_domains = len(np.unique(DataBatch))
        
        network = SCMIL_AttnMoE(model_cfg, config['ablation_params'], 
                              num_domains, run_cfg['device_encoder'], device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        
        interpret_dataset = AttnMoE_Dataset(DataList, DataLabel, DataBatch, sample_ids)
        interpret_loader = DataLoader(
            interpret_dataset, batch_size=1, shuffle=False,
            num_workers=config['training_params']['num_workers'],
            collate_fn=sparse_collate_fn
        )
        
        logger.info(f"Computing gene attributions for fold {fold_idx + 1}...")
        fold_attr = calculate_gene_attributions(network, interpret_loader, gene_list, config, device, label_map)
        sample_attributions = calculate_sample_gene_attributions(network, interpret_loader, gene_list, config, device)
        
        logger.info(f"Computing cell attentions for fold {fold_idx + 1}...")
        fold_attention_array = calculate_cell_attention(network, interpret_loader, device, config, adata)
        
        all_fold_attributions.append(fold_attr)
        all_fold_sample_attributions.append(sample_attributions)
        all_fold_attention_arrays.append(fold_attention_array)  
        
        del network
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("Ensembling results from all folds...")
    ensemble_attributions = ensemble_gene_attributions(all_fold_attributions, config)
    updated_adata = ensemble_cell_attentions(all_fold_attention_arrays, config, adata)
    
    logger.info("Generating ensemble plots...")
    plot_ensemble_gene_attributions(config, ensemble_attributions)
    plot_ensemble_cell_attention_umaps(config, updated_adata)
    ensemble_sample_results = ensemble_sample_gene_attributions(all_fold_sample_attributions, config)


    logger.info("--- Ensemble interpretability analysis completed successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCMIL Training and Interpretation Workflow")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file.")
    parser.add_argument("--cv", action="store_true", help="Run the cross-validation training.")
    parser.add_argument("--interpret", action="store_true", help="Run the ensemble interpretation analysis.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    logger = setup_logging(config)
    
    set_seed(config["run_params"]["seed"])
    logger.info(f"Global seed set to: {config['run_params']['seed']}")

    if not args.cv and not args.interpret:
        parser.error("No action requested, add --cv or --interpret.")
    
    if args.cv:
        logger.info("--- Action: Starting Cross-Validation ---")
        run_cv_experiment(args.config)
    
    if args.interpret:
        logger.info("--- Action: Starting Ensemble Interpretation ---")
        run_interpretation_ensemble(args.config)