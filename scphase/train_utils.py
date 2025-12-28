# train_utils.py
import os
import glob
import logging
import random
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import IntegratedGradients

logger = logging.getLogger(__name__)

def setup_logging(config):

    log_dir = config["path_params"]["RESULTS_DIR"]
    log_name = config["path_params"]["LOGNAME"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger("SCMIL_Workflow")

def set_seed(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_cell_attention(model, dataloader, device, config, adata):
    model.eval()
    logger.info("Calculating cell attention scores...")

    attention_scores = np.zeros(adata.n_obs)
    
    with torch.no_grad():
        for inputs, labels, _, sample_ids in tqdm(dataloader, desc="Calculating Attention"):
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            if inputs.dim() == 3 and inputs.size(0) == 1:
                inputs = inputs.squeeze(0)
            
            try:
                _, disease_output, _, attn_weights = model(inputs)
                sample_id = sample_ids[0]
                scores = attn_weights.squeeze().cpu().numpy()
                if scores.ndim == 0:
                    scores = [scores.item()]
                
                sample_mask = adata.obs[config['data_params']['sample_col']] == sample_id
                sample_indices = adata.obs.index[sample_mask]
                
                num_cells_in_sample = len(sample_indices)
                if len(scores) != num_cells_in_sample:
                    logger.warning(f"Attention scores length ({len(scores)}) doesn't match sample cell count ({num_cells_in_sample}) for sample {sample_id}")
                    min_len = min(len(scores), num_cells_in_sample)
                    scores = scores[:min_len]
                    sample_indices = sample_indices[:min_len]

                for idx, score in zip(sample_indices, scores):
                    global_pos = adata.obs.index.get_loc(idx)
                    attention_scores[global_pos] = score
                    
            except RuntimeError as e:
                logger.error(f"Error processing sample {sample_ids[0]}: {e}")
                logger.error(f"Input shape: {inputs.shape}")
                continue
    
    logger.info("Attention score calculation complete.")
    return attention_scores


def calculate_gene_attributions(model, dataloader, gene_list, config, device, label_map):
    model.eval()
    task_type = config['run_params']['task_type']
    
    def model_wrapper(inp):
        if inp.dim() == 3 and inp.shape[0] == 1:
            inp = inp.squeeze(0)
        _, disease_output, _, _ = model(inp)
        return disease_output.unsqueeze(0)
    
    ig = IntegratedGradients(forward_func=model_wrapper)

    logger.info("Extracting all samples from dataset for attribution calculation...")
    all_inputs = []
    all_labels = []
    
    for item in dataloader.dataset:
        input_tensor = item[0]
        if hasattr(input_tensor, 'is_sparse') and input_tensor.is_sparse:
            input_tensor = input_tensor.to_dense()
        input_tensor = input_tensor.float()
        
        all_inputs.append(input_tensor)
        all_labels.append(item[1].item())  

    if task_type == 'classification':
        df_attr_list = []
        unique_labels = sorted(np.unique(all_labels))
        
        for label_idx in unique_labels:
            phenotype_name = label_map.get(label_idx, f'Class_{label_idx}')
            logger.info(f"Calculating attributions for class: {phenotype_name}")
            
            class_indices = [i for i, l in enumerate(all_labels) if l == label_idx]
            class_attributions = []
            
            for i in tqdm(class_indices, desc=f"Attribution for {phenotype_name}"):
                inputs = all_inputs[i].to(device)
                if inputs.dim() == 1: 
                    inputs = inputs.unsqueeze(0)
                if inputs.shape[0] == 0: 
                    continue
                
                if hasattr(inputs, 'is_sparse') and inputs.is_sparse:
                    inputs = inputs.to_dense()
                inputs = inputs.float()
                
                try:
                    attributions = ig.attribute(inputs, target=int(label_idx), internal_batch_size=1)
                    summed_attributions = attributions.sum(dim=0).cpu().numpy()
                    class_attributions.append(summed_attributions)
                except Exception as e:
                    logger.error(f"Error calculating attribution for sample {i}: {e}")
                    logger.error(f"Input shape: {inputs.shape}, Input type: {type(inputs)}, Is sparse: {getattr(inputs, 'is_sparse', False)}")
                    continue

            if not class_attributions: 
                logger.warning(f"No samples found for class {phenotype_name}. Skipping.")
                continue
            
            mean_attributions = np.mean(class_attributions, axis=0)
            df = pd.DataFrame({
                'gene': gene_list, 
                'attribution': mean_attributions,  
                'phenotype': phenotype_name
            })
            df = df.sort_values(by="attribution", ascending=False)  
            df_attr_list.append(df)
            
        return df_attr_list
        
    elif task_type == 'regression':
        logger.info("Calculating attributions for regression task...")
        all_attributions = []
        for inputs in tqdm(all_inputs, desc="Attribution for Regression"):
            inputs = inputs.to(device)
            if inputs.dim() == 1: inputs = inputs.unsqueeze(0)
            if inputs.shape[0] == 0: continue
            
            attributions = ig.attribute(inputs, internal_batch_size=1)
            summed_attributions = attributions.sum(dim=0).cpu().numpy()
            all_attributions.append(summed_attributions)
        
        mean_attributions = np.mean(all_attributions, axis=0)
        df_features = pd.DataFrame({
            'gene': gene_list, 
            'attribution': mean_attributions  
        })
        df_features = df_features.sort_values(by='attribution', ascending=False) 
        return [df_features] 
    return []
    
    
def ensemble_gene_attributions(all_fold_attributions, config):

    logger = logging.getLogger("SCMIL_Workflow")
    
    if not all_fold_attributions:
        logger.warning("No fold attributions provided for ensemble.")
        return pd.DataFrame()
    
    label_map = config.get('label_map', None)
    task_type = config['run_params']['task_type']
    
    if task_type == 'classification':
        all_phenotypes = set()
        for fold_attr_list in all_fold_attributions:
            for df in fold_attr_list:
                if 'phenotype' in df.columns:
                    all_phenotypes.update(df['phenotype'].unique())
        
        ensemble_results = []
        
        for phenotype in sorted(all_phenotypes):
            phenotype_data = []
            for fold_attr_list in all_fold_attributions:
                for df in fold_attr_list:
                    if 'phenotype' in df.columns:
                        phenotype_df = df[df['phenotype'] == phenotype]
                        if not phenotype_df.empty:
                            phenotype_data.append(phenotype_df)
            
            if not phenotype_data:
                continue

            combined_data = pd.concat(phenotype_data, ignore_index=True)

            gene_stats = combined_data.groupby('gene').agg({
                'attribution': ['mean', 'std', 'count']
            }).reset_index()
            
            gene_stats.columns = ['gene', 'mean_attribution', 'std_attribution', 'count_attribution']
            gene_stats['phenotype'] = phenotype

            gene_stats['ci_lower'] = gene_stats['mean_attribution'] - 1.96 * gene_stats['std_attribution'] / np.sqrt(gene_stats['count_attribution'])
            gene_stats['ci_upper'] = gene_stats['mean_attribution'] + 1.96 * gene_stats['std_attribution'] / np.sqrt(gene_stats['count_attribution'])
            
            ensemble_results.append(gene_stats)
        
        if not ensemble_results:
            logger.warning("No ensemble results generated.")
            return pd.DataFrame()

        final_ensemble = pd.concat(ensemble_results, ignore_index=True)
        results_dir = config['path_params']['RESULTS_DIR']
 
        for phenotype in sorted(final_ensemble['phenotype'].unique()):
            phenotype_data = final_ensemble[final_ensemble['phenotype'] == phenotype]
            
            safe_phenotype_name = "".join(c for c in phenotype if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_phenotype_name = safe_phenotype_name.replace(' ', '_')
            filename = f"ensemble_gene_attributions_{safe_phenotype_name}.csv"
            
            filepath = os.path.join(results_dir, filename)
            phenotype_data.to_csv(filepath, index=False)
            logger.info(f"Saved ensemble gene attributions for {phenotype} to {filepath}")
    
    else:
        all_regression_data = []
        for fold_attr_list in all_fold_attributions:
            for df in fold_attr_list:
                all_regression_data.append(df)
        
        if not all_regression_data:
            logger.warning("No regression data found.")
            return pd.DataFrame()
        
        combined_data = pd.concat(all_regression_data, ignore_index=True)
        
        gene_stats = combined_data.groupby('gene').agg({
            'attribution': ['mean', 'std', 'count']
        }).reset_index()
        
        gene_stats.columns = ['gene', 'mean_attribution', 'std_attribution', 'count_attribution']

        gene_stats['ci_lower'] = gene_stats['mean_attribution'] - 1.96 * gene_stats['std_attribution'] / np.sqrt(gene_stats['count_attribution'])
        gene_stats['ci_upper'] = gene_stats['mean_attribution'] + 1.96 * gene_stats['std_attribution'] / np.sqrt(gene_stats['count_attribution'])
        
        final_ensemble = gene_stats

        results_dir = config['path_params']['RESULTS_DIR']
        filepath = os.path.join(results_dir, "ensemble_gene_attributions_regression.csv")
        final_ensemble.to_csv(filepath, index=False)
        logger.info(f"Saved ensemble gene attributions for regression to {filepath}")
    
    return final_ensemble


def ensemble_cell_attentions(all_fold_attention_arrays, config, adata):

    logger = logging.getLogger("SCMIL_Workflow")
    logger.info("Ensembling cell attentions from all folds...")
    
    if not all_fold_attention_arrays:
        logger.warning("No fold attention arrays provided for ensemble.")
        return adata
    
    stacked_arrays = np.stack(all_fold_attention_arrays, axis=0)
    ensemble_attention_scores = np.mean(stacked_arrays, axis=0)
    
    adata.obs['attention_weight_mean'] = ensemble_attention_scores

    attention_std = np.std(stacked_arrays, axis=0)
    adata.obs['attention_weight_std'] = attention_std

    output_path = os.path.join(config['path_params']['RESULTS_DIR'], 'ensemble_adata_with_attention.h5ad')
    adata.write_h5ad(output_path, compression='gzip')
    logger.info(f"Saved ensemble adata with attention weights to {output_path}")
    
 
    sample_col = config['data_params']['sample_col'] 
    attention_summary = pd.DataFrame({
        'cell_barcode': adata.obs.index,
        'sample_id': adata.obs[sample_col],  
        'attention_weight_mean': ensemble_attention_scores,
        'attention_weight_std': attention_std
    })
    
    csv_output_path = os.path.join(config['path_params']['RESULTS_DIR'], 'ensemble_cell_attentions_summary.csv')
    attention_summary.to_csv(csv_output_path, index=False)
    logger.info(f"Saved ensemble cell attention summary to {csv_output_path}")
    
    return adata


def plot_ensemble_gene_attributions(config, ensemble_attributions):

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    
    logger = logging.getLogger("SCMIL_Workflow")
    
    if ensemble_attributions.empty:
        logger.warning("No ensemble attributions to plot.")
        return
    
    results_dir = config['path_params']['RESULTS_DIR']
    task_type = config['run_params']['task_type']

    color_schemes = {
        'green_gradient': ['#E8F5E8', '#90EE90', '#32CD32', '#228B22', '#006400'],  
        'blue_gradient': ['#E6F3FF', '#87CEEB', '#4169E1', '#1E90FF', '#0000CD'],  
        'orange_gradient': ['#FFF8DC', '#FFE4B5', '#FFA500', '#FF8C00', '#FF4500'], 
        'purple_gradient': ['#F3E5F5', '#CE93D8', '#9C27B0', '#7B1FA2', '#4A148C'],
        'red_gradient': ['#FFEBEE', '#FFCDD2', '#F44336', '#D32F2F', '#B71C1C'],  
        'teal_gradient': ['#E0F2F1', '#80CBC4', '#26A69A', '#00796B', '#004D40'],  
        'pink_gradient': ['#FCE4EC', '#F8BBD9', '#E91E63', '#C2185B', '#880E4F'],   
        'brown_gradient': ['#EFEBE9', '#BCAAA4', '#8D6E63', '#5D4037', '#3E2723'],
    }
    
    scheme_names = list(color_schemes.keys())
    
    if task_type == 'classification':
        phenotypes = sorted(ensemble_attributions['phenotype'].unique())
        
        for i, phenotype in enumerate(phenotypes):
            phenotype_data = ensemble_attributions[ensemble_attributions['phenotype'] == phenotype]
            top_genes = phenotype_data.nlargest(20, 'mean_attribution')
            plt.figure(figsize=(10, 8))
            top_genes_reversed = top_genes.iloc[::-1]
            current_scheme = color_schemes[scheme_names[i % len(scheme_names)]]
            attribution_values = top_genes_reversed['mean_attribution'].values
            normalized_values = (attribution_values - attribution_values.min()) / (attribution_values.max() - attribution_values.min() + 1e-8)
            cmap = mcolors.LinearSegmentedColormap.from_list(f"custom_{phenotype}", current_scheme)
            colors = cmap(normalized_values)
            bars = plt.barh(range(len(top_genes_reversed)), top_genes_reversed['mean_attribution'], 
                           color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            plt.yticks(range(len(top_genes_reversed)), top_genes_reversed['gene'])
            plt.xlabel('Mean Attribution Score', fontsize=12, fontweight='bold')
            plt.ylabel('Genes', fontsize=12, fontweight='bold')
            plt.title(f'Top 20 Genes - {phenotype}', fontsize=14, fontweight='bold')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=attribution_values.min(), vmax=attribution_values.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
            cbar.set_label('Attribution Score', fontsize=10, fontweight='bold')
            plt.tight_layout()
            
            safe_phenotype_name = "".join(c for c in phenotype if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_phenotype_name = safe_phenotype_name.replace(' ', '_')
            filename = f"ensemble_gene_attributions_{safe_phenotype_name}.pdf"
            
            filepath = os.path.join(results_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved gene attribution plot for {phenotype} to {filepath} using {scheme_names[i % len(scheme_names)]} color scheme")
    
    else:
        top_genes = ensemble_attributions.nlargest(20, 'mean_attribution')
        plt.figure(figsize=(10, 8))
        top_genes_reversed = top_genes.iloc[::-1]
        current_scheme = color_schemes['blue_gradient']
        attribution_values = top_genes_reversed['mean_attribution'].values
        normalized_values = (attribution_values - attribution_values.min()) / (attribution_values.max() - attribution_values.min() + 1e-8)

        cmap = mcolors.LinearSegmentedColormap.from_list("custom_regression", current_scheme)
        colors = cmap(normalized_values)
        
        bars = plt.barh(range(len(top_genes_reversed)), top_genes_reversed['mean_attribution'], 
                       color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        plt.yticks(range(len(top_genes_reversed)), top_genes_reversed['gene'])
        plt.xlabel('Mean Attribution Score', fontsize=12, fontweight='bold')
        plt.ylabel('Genes', fontsize=12, fontweight='bold')
        plt.title('Top 20 Genes - Regression', fontsize=14, fontweight='bold')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=attribution_values.min(), vmax=attribution_values.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label('Attribution Score', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(results_dir, "ensemble_gene_attributions_regression.pdf")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved gene attribution plot for regression to {filepath}")


def plot_ensemble_cell_attention_umaps(config, adata_with_attention):

    logger = logging.getLogger("SCMIL_Workflow")
    logger.info("Plotting ensemble cell attention UMAPs...")
    
    results_dir = config['path_params']['RESULTS_DIR']
    
    label_col = config.get('data_params', {}).get('label_col', 'label')
    sample_col = config.get('data_params', {}).get('sample_col', 'sample_id')

    if hasattr(adata_with_attention, 'obs'):
        adata = adata_with_attention.copy()
        logger.info("Using AnnData object directly for plotting")
        
        if 'attention_weight_mean' not in adata.obs.columns:
            logger.error("No attention_weight_mean found in AnnData.obs")
            return
        
        adata.obs['attn'] = adata.obs['attention_weight_mean']

        adata.obs['attn_scaled'] = np.nan
        for sample_id in adata.obs[sample_col].unique():
            sample_mask = adata.obs[sample_col] == sample_id
            sample_attn = adata.obs.loc[sample_mask, 'attn']
            
            if len(sample_attn) > 0 and sample_attn.sum() > 0:
                avg_score = 1 / len(sample_attn)
                log_attn = np.log2(sample_attn / avg_score + 1e-9)
                attn_scaled = (log_attn - np.mean(log_attn)) / (np.std(log_attn) + 1e-9)
                attn_scaled_clipped = np.clip(attn_scaled, -1, 1)
                adata.obs.loc[sample_mask, 'attn_scaled'] = attn_scaled_clipped
        
        adata.obs['attn_scaled'] = adata.obs['attn_scaled'].fillna(0)
        
    else:
        logger.info("Processing DataFrame object for plotting")
        adata = sc.read_h5ad(config['path_params']['data_h5ad_file'])
        attention_scores = np.zeros(adata.n_obs)
        for _, row in adata_with_attention.iterrows():
            sample_id = row[sample_col]
            cell_idx = int(row['cell_index_in_sample'])
            attention_mean = row['attention_weight']
            sample_mask = adata.obs[sample_col] == sample_id
            sample_cells = adata.obs[sample_mask]
            
            if cell_idx < len(sample_cells):
                global_idx = sample_cells.index[cell_idx]
                attention_scores[adata.obs.index.get_loc(global_idx)] = attention_mean

        adata.obs['attn'] = attention_scores

        adata.obs['attn_scaled'] = np.nan
        for sample_id in adata.obs[sample_col].unique():
            sample_mask = adata.obs[sample_col] == sample_id
            sample_attn = adata.obs.loc[sample_mask, 'attn']
            
            if len(sample_attn) > 0 and sample_attn.sum() > 0:
                avg_score = 1 / len(sample_attn)
                log_attn = np.log2(sample_attn / avg_score + 1e-9)
                attn_scaled = (log_attn - np.mean(log_attn)) / (np.std(log_attn) + 1e-9)
                attn_scaled_clipped = np.clip(attn_scaled, -1, 1)
                adata.obs.loc[sample_mask, 'attn_scaled'] = attn_scaled_clipped
        
        adata.obs['attn_scaled'] = adata.obs['attn_scaled'].fillna(0)
    
    task_type = config.get('run_params', {}).get('task_type', 'classification')
    
    sc.settings.figdir = results_dir
    
    if task_type == 'classification':
        if 'celltype' in adata.obs.columns:
            sc.pl.umap(adata, color='celltype',show=False,palette=sns.color_palette("husl", 24),legend_fontsize=6,frameon=True,title='Cell Type',save='_celltype.pdf')
            sc.pl.umap(adata, color='celltype',show=False,palette=sns.color_palette("husl", 24),legend_fontsize=6,legend_loc = 'on data', frameon=True,title='Cell Type',save='_celltype_ondata.pdf')

        sc.pl.umap(adata, color='attn_scaled',show=False,cmap='RdYlBu_r',frameon=True,title='Ensemble Cell Attention (All)',save='_ensemble_attention_all.pdf')
        
        if label_col in adata.obs.columns:
            unique_labels = sorted(adata.obs[label_col].unique())
            logger.info(f"Found {len(unique_labels)} unique labels in '{label_col}' column: {unique_labels}")
            
            for label in unique_labels:
                label_mask = adata.obs[label_col] == label
                adata_label = adata[label_mask].copy()
                
                logger.info(f"Plotting attention for {label_col} {label} ({adata_label.n_obs} cells)")
                sc.pl.umap(adata_label, color='attn_scaled',show=False,cmap='RdYlBu_r',frameon=True,title=f'Cell Attention - {label_col.capitalize()} {label}',save=f'_attention_{label_col}_{label}.pdf')
                
        else:
            logger.warning(f"No '{label_col}' column found in adata.obs, cannot create group-specific plots")
            logger.info(f"Available columns in adata.obs: {list(adata.obs.columns)}")
    
    else:
        sc.pl.umap(adata, color='attn_scaled',show=False,cmap='RdYlBu_r',frameon=True,title='Ensemble Cell Attention (Regression)',save='_ensemble_attention_regression.pdf')
    
    logger.info(f"Cell attention UMAP plots saved to {results_dir}")



def calculate_sample_gene_attributions(model, dataloader, gene_list, config, device):

    import gc
    
    model.eval()
    task_type = config['run_params']['task_type']
    
    logger.info("Starting sample-level attribution calculation using original PHASE method...")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, task_type):
            super().__init__()
            self.model = model
            self.task_type = task_type
        
        def forward(self, x):
            output = self.model(x)
            if isinstance(output, tuple):
                output = output[0]
            
            if self.task_type == 'classification':
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                elif output.dim() == 3:
                    output = output.squeeze(1) if output.shape[1] == 1 else output.view(output.shape[0], -1)
            else:
                if output.dim() > 2:
                    output = output.view(output.shape[0], -1)
            
            return output

    wrapped_model = ModelWrapper(model, task_type)
    ig = IntegratedGradients(wrapped_model)
    logger.info("IG initialized successfully with model wrapper")
    
    long_format_data = []
    
    for i, data in enumerate(tqdm(dataloader, desc="Sample Attribution")):
        try:
            if isinstance(data, (list, tuple)) and len(data) == 4:
                inputs = data[0]
                labels = data[1] 
                batch_labels = data[2]  
                batch_sample_ids = data[3]  
            else:
                logger.error(f"Unexpected data format at batch {i}: expected 4 elements, got {len(data) if isinstance(data, (list, tuple)) else 'non-sequence'}")
                continue

            if not isinstance(inputs, torch.Tensor):
                logger.error(f"Inputs is not a tensor at batch {i}")
                continue
                
            if not isinstance(labels, torch.Tensor):
                if isinstance(labels, (int, float)):
                    labels = torch.tensor([labels])
                else:
                    logger.error(f"Cannot convert labels to tensor at batch {i}")
                    continue
            
            labels = labels.long()
            
            if hasattr(inputs, 'is_sparse') and inputs.is_sparse:
                inputs = inputs.to_dense()
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            
            if inputs.dim() == 3 and inputs.shape[0] == 1:
                inputs = inputs.squeeze(0) 
            elif inputs.dim() == 1:
                inputs = inputs.unsqueeze(0) 
            
            logger.debug(f"Batch {i}: inputs shape {inputs.shape}, labels shape {labels.shape}")
       
            if task_type == 'classification':
                target_label = labels.item() if labels.numel() == 1 else labels[0].item()
                ig_attr_test = ig.attribute(inputs=inputs, target=target_label, n_steps=10)
            else:
                ig_attr_test = ig.attribute(inputs=inputs, n_steps=10)
            
            inputs.detach()
            labels.detach()
            
            logger.debug(f"Attribution shape: {ig_attr_test.size()}")
    
            ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
            logger.debug(f"Summed attribution shape: {ig_attr_test_sum.shape}")

            if len(ig_attr_test_sum) != len(gene_list):
                logger.warning(f"Attribution length {len(ig_attr_test_sum)} doesn't match gene list length {len(gene_list)} for batch {i}")
                if len(ig_attr_test_sum) > len(gene_list):
                    ig_attr_test_sum = ig_attr_test_sum[:len(gene_list)]
                else:
                    padded_attr = np.zeros(len(gene_list))
                    padded_attr[:len(ig_attr_test_sum)] = ig_attr_test_sum
                    ig_attr_test_sum = padded_attr

            ig_attr_test_norm_sum_1 = (ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)).copy()
            
            sample_id = batch_sample_ids[0] if isinstance(batch_sample_ids, list) else str(batch_sample_ids)
                
            label_value = target_label if task_type == 'classification' else labels.cpu().item()

            for gene_idx, (gene_name, attribution_score) in enumerate(zip(gene_list, ig_attr_test_norm_sum_1)):
                long_format_data.append({
                    'sample_id': sample_id,  
                    'gene': gene_name,
                    'attribution_score': attribution_score,
                    'label': label_value
                })
            
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error calculating attribution for batch {i}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    if not long_format_data:
        logger.warning("No attributions calculated.")
        return pd.DataFrame()

    df_long = pd.DataFrame(long_format_data)
    
    logger.info(f"Attribution calculation completed. Long format shape: {df_long.shape}")
    logger.info('Sample-level IG attribution calculation finished!')
    
    return df_long


def ensemble_sample_gene_attributions(all_fold_sample_attributions, config):
    logger = logging.getLogger("SCMIL_Workflow")
    
    if not all_fold_sample_attributions:
        logger.warning("No fold sample attributions provided for ensemble.")
        return pd.DataFrame()
    
    valid_fold_attributions = [df for df in all_fold_sample_attributions if not df.empty]
    
    if not valid_fold_attributions:
        logger.warning("No valid fold sample attributions found.")
        return pd.DataFrame()
    
    logger.info(f"Ensembling sample attributions from {len(valid_fold_attributions)} folds...")

    all_data = pd.concat(valid_fold_attributions, ignore_index=True)

    unique_genes = sorted(all_data['gene'].unique())
    unique_samples = sorted(all_data['sample_id'].unique())
    
    logger.info(f"Processing {len(unique_genes)} genes")
    logger.info(f"Processing {len(unique_samples)} unique samples")
    
    ensemble_data = all_data.groupby(['sample_id', 'gene'])['attribution_score'].mean().reset_index()

    wide_format = ensemble_data.pivot(index='sample_id', columns='gene', values='attribution_score')
    
    for gene in unique_genes:
        if gene not in wide_format.columns:
            wide_format[gene] = 0.0

    wide_format = wide_format[sorted(wide_format.columns)]

    results_dir = config['path_params']['RESULTS_DIR']
    mean_filepath = os.path.join(results_dir, "sample_gene_attribution_mean.csv")
    wide_format.to_csv(mean_filepath, index=True)
    logger.info(f"Saved mean attribution matrix to {mean_filepath}")
    
    logger.info(f"Successfully processed {len(unique_samples)} samples with {len(unique_genes)} genes")
    logger.info(f"Final matrix shape: {wide_format.shape} (samples × genes)")
    
    return wide_format
