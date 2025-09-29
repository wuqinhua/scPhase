# run_cv.py
import os
import json
import logging
import random
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import argparse
from scipy.stats import pearsonr

from data_loader import load_data, AttnMoE_Dataset, sparse_collate_fn
from model import SCMIL_AttnMoE
from modules import MILLoss
from train_utils import set_seed, worker_init_fn  


class EarlyStopping:
    def __init__(self, patience=15, verbose=True, delta=0, path=None, trace_func=None, mode='max'):
        self.patience, self.verbose, self.delta = patience, verbose, delta
        self.counter, self.best_score, self.early_stop = 0, None, False
        self.mode = mode
        self.path = path
        self.trace_func = trace_func if trace_func is not None else logging.info
        self.best_model_state_dict = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, is_first=True)
            return
        improved = False
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:  # mode == 'min'
            improved = score < self.best_score - self.delta
        
        if improved:
            previous_score = self.best_score
            self.best_score = score
            self.save_checkpoint(score, model, previous_score=previous_score)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} '
                          f'(current: {score:.6f}, best: {self.best_score:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model, previous_score=None, is_first=False):
        if self.verbose:
            if is_first:
                self.trace_func(f'Initial validation score: {score:.6f}. Storing best model state...')
            elif previous_score is not None:
                self.trace_func(f'Validation score improved ({previous_score:.6f} --> {score:.6f}). Storing best model state...')
        
        self.best_model_state_dict = copy.deepcopy(model.state_dict())
        if self.path:
            torch.save(self.best_model_state_dict, self.path)
            self.trace_func(f'Saving model to {self.path}')
            

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_domain_loss_weight(epoch, total_epochs):
    p = epoch / total_epochs
    return 0.2 * (1 / (1 + np.exp(-10 * (p - 0.3))))

def train_and_evaluate_fold(fold, train_idx, test_idx, DataList, DataLabel, DataBatch, SampleIDs, cfg, use_domain_adaptation):
    
    logger = logging.getLogger("SCMIL_Experiment_Runner")
    logger.info(f"--- Starting Fold {fold + 1} ---")
 
    fold_seed = cfg['run_params']['seed']  
    set_seed(fold_seed)
    logger.info(f"Set seed for fold {fold + 1}: {fold_seed}")

    run_cfg, train_cfg, optim_cfg, sched_cfg, ablation_cfg, model_cfg, path_cfg = \
        cfg['run_params'], cfg['training_params'], cfg['optimizer_params'], \
        cfg['scheduler_params'], cfg['ablation_params'], cfg['model_params'], cfg['path_params']
    
    task_type = run_cfg['task_type']
    device_model = run_cfg['device_model']
    
    # Split data for this fold
    X_train_full, y_train_full, batch_train_full = [DataList[i] for i in train_idx], DataLabel[train_idx], DataBatch[train_idx]
    X_test, y_test, batch_test = [DataList[i] for i in test_idx], DataLabel[test_idx], DataBatch[test_idx]
    logger.info(f"Test set: {len(X_test)} samples from group {np.unique(batch_test)}.")
    
    # Create validation set
    stratify_val = y_train_full if task_type == 'classification' else None
    train_indices, val_indices = train_test_split(np.arange(len(X_train_full)), test_size=train_cfg['val_size'], random_state=cfg['run_params']['seed'], stratify=stratify_val)
    
    train_data, train_label, train_batch = [X_train_full[i] for i in train_indices], y_train_full[train_indices], batch_train_full[train_indices]
    valid_data, valid_label, valid_batch = [X_train_full[i] for i in val_indices], y_train_full[val_indices], batch_train_full[val_indices]

    logger.info(f"Train set: {len(train_data)} samples. Validation set: {len(valid_data)} samples. Test set: {len(X_test)} samples.")
    if task_type == 'classification':
        logger.info(f"Train labels: {np.bincount(train_label)}. Validation labels: {np.bincount(valid_label)}. Test labels: {np.bincount(y_test)}.")

    unique_domains = np.unique(batch_train_full)
    domain_mapping = {domain: idx for idx, domain in enumerate(unique_domains)}
    train_batch_mapped = np.array([domain_mapping[b] for b in train_batch])
    valid_batch_mapped = np.array([domain_mapping[b] for b in valid_batch])
    
    network = SCMIL_AttnMoE(model_cfg, ablation_cfg, len(unique_domains), run_cfg['device_encoder'], device_model)

    train_sample_ids = [SampleIDs[i] for i in train_indices]
    val_sample_ids = [SampleIDs[i] for i in val_indices]
    sample_ids_test = [SampleIDs[i] for i in test_idx]
    
    trainData = AttnMoE_Dataset(train_data, train_label, train_batch_mapped, train_sample_ids)
    validData = AttnMoE_Dataset(valid_data, valid_label, valid_batch_mapped, val_sample_ids)
    trainLoader = DataLoader(
        trainData, 
        batch_size=train_cfg['batch_size'], 
        num_workers=train_cfg['num_workers'], 
        collate_fn=sparse_collate_fn, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )
    validLoader = DataLoader(
        validData, 
        batch_size=train_cfg['batch_size'], 
        num_workers=train_cfg['num_workers'], 
        collate_fn=sparse_collate_fn, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )

    # Initialize loss function based on task
    if task_type == 'classification':
        class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_label), y=train_label), dtype=torch.float).to(device_model)
        disease_criterion = MILLoss(class_weights=class_weights)
        early_stopping_mode = 'max' # Maximize AUC
    else: # regression
        disease_criterion = nn.MSELoss()
        early_stopping_mode = 'max' # Maximize R2 instead of minimizing loss
    
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(network.parameters(), lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'], betas=tuple(optim_cfg['betas']))
    scheduler = get_cosine_schedule_with_warmup(optimizer, sched_cfg['warmup_epochs'] * len(trainLoader), train_cfg['epochs'] * len(trainLoader))
    
    checkpoint_path = os.path.join(path_cfg['RESULTS_DIR'], f"BestModel_{path_cfg['MODEL_NAME']}_Fold{fold+1}.pt")
    early_stopping = EarlyStopping(patience=train_cfg['early_stopping_patience'], verbose=True, path=checkpoint_path, mode=early_stopping_mode)

    # Training loop
    for epoch in range(train_cfg['epochs']):
        network.train()
        total_train_loss, total_domain_loss = 0.0, 0.0
        
        pbar = tqdm(trainLoader, desc=f'Epoch {epoch+1}/{train_cfg["epochs"]} Train')
        optimizer.zero_grad()  
        for d, l, b, _ in pbar:
            d, l, b = d[0].float().to(device_model), l.to(device_model), b.to(device_model)
            if task_type == 'classification':
                l = l.long()
            else:
                l = l.float().unsqueeze(0)

            alpha = 2. / (1. + np.exp(-8 * float(epoch) / train_cfg['epochs'])) - 1
            domain_weight = get_domain_loss_weight(epoch, train_cfg['epochs']) if use_domain_adaptation else 0.0
            
            _, disease_out, domain_out, _ = network(d, alpha=alpha)
            
            disease_loss = disease_criterion(disease_out.unsqueeze(0), l)
            loss = disease_loss

            if domain_weight > 0 and domain_out is not None:
                domain_loss = domain_criterion(domain_out.unsqueeze(0), b.long())
                loss += domain_loss * domain_weight
                total_domain_loss += domain_loss.item()
            else:
                domain_loss = torch.tensor(0.0).to(device_model)
                total_domain_loss += 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), optim_cfg['clip_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_train_loss += disease_loss.item()
    
        # Validation loop
        network.eval()
        val_loss, val_preds, val_probs, val_true = 0.0, [], [], []
        with torch.no_grad():
            for d, l, _, _ in tqdm(validLoader, desc=f'Epoch {epoch+1}/{train_cfg["epochs"]} Valid'):
                d, l = d[0].float().to(device_model), l.to(device_model)
                if task_type == 'classification':
                    l = l.long()
                else:
                    l = l.float().unsqueeze(0)

                _, disease_out, _, _ = network(d, alpha=1.0)
                val_loss += disease_criterion(disease_out.unsqueeze(0), l).item()
                
                if task_type == 'classification':
                    probs = F.softmax(disease_out, dim=0)
                    val_preds.append(torch.argmax(probs).cpu().numpy())
                    val_probs.append(probs.cpu().numpy())
                    val_true.append(l.cpu().numpy())
                else: # regression
                    val_preds.append(disease_out.cpu().numpy())
                    val_true.append(l.squeeze().cpu().numpy())
        
        # Log metrics and check for early stopping
        avg_val_loss = val_loss / len(validLoader)
        if task_type == 'classification':
            val_auc = calculate_auc_score(val_true, val_probs, model_cfg['n_classes'])
            val_acc = accuracy_score(val_true, val_preds)
            logger.info(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(trainLoader):.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f} | Domain Loss: {total_domain_loss/len(trainLoader):.4f}")
            early_stopping(val_auc, network)
        else: # regression
            val_r2 = r2_score(val_true, val_preds)
            logger.info(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(trainLoader):.4f} | Val Loss: {avg_val_loss:.4f} | Val R2: {val_r2:.4f}")
            early_stopping(val_r2, network)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break
            
    # Load best model and evaluate on the test set
    logger.info(f"Loading best model from {checkpoint_path} for final test evaluation.")
    network.load_state_dict(torch.load(checkpoint_path))
    network.eval()
    
    testData = AttnMoE_Dataset(X_test, y_test, batch_test, sample_ids_test)
    testLoader = DataLoader(testData, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], collate_fn=sparse_collate_fn)
    
    test_preds, test_probs, test_true = [], [], []
    with torch.no_grad():
        for d, l, _, _ in testLoader:
            d = d[0].float().to(device_model)
            _, disease_out, _, _ = network(d, alpha=1.0)

            if task_type == 'classification':
                probs = F.softmax(disease_out, dim=0)
                test_preds.append(torch.argmax(probs).cpu().numpy())
                test_probs.append(probs.cpu().numpy())
            else: # regression
                test_preds.append(disease_out.cpu().numpy().squeeze())
            test_true.append(l.numpy().squeeze())

    # Calculate final metrics
    results = {}
    if task_type == 'classification':
        results['auc'] = calculate_auc_score(test_true, test_probs, model_cfg['n_classes'])
        results['acc'] = accuracy_score(test_true, test_preds)
        results['precision'] = precision_score(test_true, test_preds, average='weighted', zero_division=0)
        results['recall'] = recall_score(test_true, test_preds, average='weighted', zero_division=0)
        results['f1'] = f1_score(test_true, test_preds, average='weighted', zero_division=0)
        logger.info(f"Fold {fold+1} Test Results: AUC={results['auc']:.4f}, ACC={results['acc']:.4f}, F1={results['f1']:.4f}")
        
        # Save prediction data for this fold
        _save_fold_prediction_data(
            y_true=np.array(test_true),
            y_pred=np.array(test_preds), 
            y_prob=np.array(test_probs),
            sample_ids=sample_ids_test,
            model_name=cfg['path_params']['MODEL_NAME'],
            fold_idx=fold + 1,
            test_group=str(np.unique(batch_test)),
            auc_score=results['auc']
        )
    else: # regression
        results['mse'] = mean_squared_error(test_true, test_preds)
        results['mae'] = mean_absolute_error(test_true, test_preds)
        results['r2'] = r2_score(test_true, test_preds)
        results['person'] = pearsonr(test_true, test_preds)[0]
        logger.info(f"Fold {fold+1} Test Results: MSE={results['mse']:.4f}, MAE={results['mae']:.4f}, R2={results['r2']:.4f}, Person={results['person']:.4f}")
        
        # Save prediction data for this fold
        _save_fold_prediction_data(
            y_true=np.array(test_true),
            y_pred=np.array(test_preds), 
            y_prob=None,
            sample_ids=sample_ids_test,
            model_name=cfg['path_params']['MODEL_NAME'],
            fold_idx=fold + 1,
            test_group=str(np.unique(batch_test)),
            auc_score=results['person']
        )
        
    return results


def _save_fold_prediction_data(y_true, y_pred, y_prob, sample_ids, model_name, fold_idx, test_group, auc_score):
    n_samples = len(y_true)
    if not hasattr(_save_fold_prediction_data, 'prediction_storage'):
        _save_fold_prediction_data.prediction_storage = []

    for i in range(n_samples):
        sample_data = {
            'model_name': model_name,
            'fold': fold_idx,
            'test_group': test_group,
            'sample_idx': i,
            'y_true': y_true[i],
            'y_pred': y_pred[i],
            'auc_score': auc_score
        }

        if y_prob is not None:
            for class_idx in range(y_prob.shape[1]):
                sample_data[f'prob_class_{class_idx}'] = y_prob[i, class_idx]
        
        _save_fold_prediction_data.prediction_storage.append(sample_data)
        
        
def run_cv_experiment(config_path="config.json"):

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger = logging.getLogger("SCMIL_Workflow")
    logger.info("--- Starting New CV Experiment Run ---")
    logger.info(f"Loaded configuration from: {config_path}")
    logger.info(f"Configuration: \n{json.dumps(config, indent=2)}")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Forcing CPU.")
        config["run_params"]["device_model"] = "cpu"
        config["run_params"]["device_encoder"] = "cpu"
    
    if hasattr(_save_fold_prediction_data, 'prediction_storage'):
        _save_fold_prediction_data.prediction_storage.clear()
    
    logger.info("Loading data...")
    DataList, DataLabel, DataBatch, SampleIDs = load_data(config)
    logger.info(f"Data loading complete. Total samples: {len(DataList)}")
    
    num_groups = len(np.unique(DataBatch))
    task_type = config['run_params']['task_type']

    # 1. Automatic Domain Adaptation
    if num_groups > 1 and task_type == 'classification':
        use_domain_adaptation = True
        logger.info(f"Detected {num_groups} groups for a classification task. Enabling Domain Adaptation.")
    else:
        use_domain_adaptation = False
        if num_groups <= 1:
            logger.info("Single data group detected. Disabling Domain Adaptation.")
        if task_type == 'regression':
            logger.info("Regression task detected. Disabling classifier-based Domain Adaptation.")
            
    # 2. Select CV Strategy
    if num_groups > 1:
        logger.info("Using Leave-One-Group-Out (LOGO) Cross-Validation.")
        cv = LeaveOneGroupOut()
        cv_splitter = cv.split(DataList, DataLabel, DataBatch)
    else:
        num_folds = config['run_params']['num_folds']
        logger.info(f"Using {num_folds}-Fold Cross-Validation.")
        if task_type == 'classification':
            cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config['run_params']['seed'])
            cv_splitter = cv.split(DataList, DataLabel)
        else: # regression
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=config['run_params']['seed'])
            cv_splitter = cv.split(DataList)

    all_results = []
    
    # --- CROSS-VALIDATION LOOP ---
    for fold, (train_idx, test_idx) in enumerate(cv_splitter):
        test_group_info = np.unique(DataBatch[test_idx]) if num_groups > 1 else f"Fold {fold+1}"
        
        # Check if skip_groups parameter exists and skip specified groups
        if 'skip_groups' in config['run_params'] and config['run_params']['skip_groups']:
            skip_groups = config['run_params']['skip_groups']
            if num_groups > 1:
                # For LOGO CV, check if test group should be skipped
                test_group = test_group_info[0] if isinstance(test_group_info, np.ndarray) else test_group_info
                if test_group in skip_groups:
                    logger.info(f"Skipping fold {fold+1} with test group {test_group_info} as specified in skip_groups: {skip_groups}")
                    continue
            else:
                # For K-fold CV, check if fold number should be skipped
                if (fold + 1) in skip_groups:
                    logger.info(f"Skipping fold {fold+1} as specified in skip_groups: {skip_groups}")
                    continue
        else:
            # Fallback to original logic: automatically skip fold if the test batch doesn't contain all classes for classification tasks
            if num_groups > 1 and task_type == 'classification':
                n_classes_in_fold = len(np.unique(DataLabel[test_idx]))
                total_classes = config['model_params']['n_classes']
                if n_classes_in_fold < total_classes:
                    logger.info(f"Skipping fold {fold+1} with test group {test_group_info} because it only contains {n_classes_in_fold}/{total_classes} classes.")
                    continue

        fold_raw_results = train_and_evaluate_fold(fold, train_idx, test_idx, DataList, DataLabel, DataBatch, SampleIDs, config, use_domain_adaptation)
        
        fold_with_meta = {
            "model_name": config['path_params']['MODEL_NAME'],
            "fold": fold + 1,
            "test_group": str(test_group_info),
            **fold_raw_results
        }
        all_results.append(fold_with_meta)

    if not all_results:
        logger.warning("No folds were executed. Check configuration and data.")
        return

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(config['path_params']['RESULTS_DIR'], f"AllFolds_{config['path_params']['MODEL_NAME']}.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nFull results for all folds saved to: {results_path}")

    _save_predictions_csv(config)

    summary_metrics = [col for col in results_df.columns if col not in ['model_name', 'fold', 'test_group']]
    summary = {'model_name': config['path_params']['MODEL_NAME']}
    logger.info("\n--- FINAL EXPERIMENT SUMMARY ---")
    for metric in summary_metrics:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        summary[f'mean_{metric}'] = mean_val
        summary[f'std_{metric}'] = std_val
        logger.info(f"Mean {metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")

    summary_path = os.path.join(config['path_params']['RESULTS_DIR'], f"Summary_{config['path_params']['MODEL_NAME']}.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    logger.info(f"\nSummary results saved to: {summary_path}")
    logger.info("--- CV Experiment finished successfully! ---")

def _save_predictions_csv(config):
    logger = logging.getLogger("SCMIL_Workflow")
    if not hasattr(_save_fold_prediction_data, 'prediction_storage') or not _save_fold_prediction_data.prediction_storage:
        logger.warning("No prediction data to save")
        return
    
    try:
        predictions_dir = os.path.join(config['path_params']['RESULTS_DIR'], 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)

        df = pd.DataFrame(_save_fold_prediction_data.prediction_storage)

        model_name = config['path_params']['MODEL_NAME']
        model_filename = f'{model_name}_predictions.csv'
        model_filepath = os.path.join(predictions_dir, model_filename)
        df.to_csv(model_filepath, index=False)
        
        all_filepath = os.path.join(predictions_dir, 'all_models_predictions.csv')
        df.to_csv(all_filepath, index=False)
        
        logger.info(f"Predictions saved to: {model_filepath}")
        logger.info(f"Predictions also saved as: {all_filepath}")
        logger.info(f"Saved {len(df)} predictions across {df['fold'].nunique()} folds")
        
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")


def calculate_auc_score(y_true, y_probs, n_classes):
    y_true_array = np.array(y_true)
    y_probs_array = np.array(y_probs)
    
    actual_classes = len(np.unique(y_true_array))
    
    if actual_classes == 2 or n_classes == 2:
        if y_probs_array.ndim == 2 and y_probs_array.shape[1] == 2:
            return roc_auc_score(y_true_array, y_probs_array[:, 1])
        else:
            return roc_auc_score(y_true_array, y_probs_array)
    else:
        return roc_auc_score(y_true_array, y_probs_array, multi_class='ovr', labels=np.arange(n_classes))
