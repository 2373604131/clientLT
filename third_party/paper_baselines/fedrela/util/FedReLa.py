# -*- coding: utf-8 -*-
"""
FedReLa: Federated Relabeling Module
This module provides functions for relabeling local data in federated learning scenarios.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F


def collect_Q(g_head, net, train_dataloader, args, method='loge', num_epochs=5):
    """
    Collect prediction probabilities Q over multiple epochs.
    
    Args:
        g_head: Global head classifier
        net: Network backbone
        train_dataloader: Training data loader
        args: Arguments object
        method: Method for computing logits ('loge' or 'etf')
        num_epochs: Number of epochs to collect predictions
        
    Returns:
        pred_dict: Dictionary mapping sample indices to (averaged_prob, true_label)
        keys: List of sample index keys
        targets: Array of true labels
        preds: Array of averaged prediction probabilities
    """
    pred_dict = {}
    g_head.eval()
    
    for e in range(num_epochs):
        for batch_idx, (index, x, target) in enumerate(train_dataloader):
            index, x, target = index, x.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            feat = net(x, latent_output=True)
            out = g_head(feat)
            if method == 'etf':
                cur_M = g_head.ori_M
                out = torch.matmul(out, cur_M)
            pospre = F.softmax(out, dim=1).cpu().detach().numpy()
            target = target.long()
            
            for i, value in enumerate(x.cpu()):
                if str(index[i]) in pred_dict:
                    pred_dict[str(index[i])].append((pospre[i], target.cpu().detach().numpy()[i]))
                else:
                    pred_dict[str(index[i])] = [(pospre[i], target.cpu().detach().numpy()[i])]
    
    # Average predictions for each sample
    targets = np.array([t[0][1] for t in pred_dict.values()])
    for k, v in pred_dict.items():
        pred_dict[k] = (np.mean(np.array([item[0] for item in v]), axis=0), v[0][1])
    
    preds = np.array([t[0] for t in pred_dict.values()])
    keys = list(pred_dict.keys())
    
    return pred_dict, keys, targets, preds


def compute_normalized_priors(targets, class_num):
    """
    Compute normalized priors based on class distribution.
    
    Args:
        targets: Array of true labels
        class_num: Number of classes
        
    Returns:
        priors: Normalized prior weights for each class
        cls_cnt_full: Full class count array
    """
    unique_cls, cls_counts = np.unique(targets, return_counts=True)
    print(unique_cls, cls_counts)
    
    # Build full class count array
    cls_cnt_full = np.zeros(class_num, dtype=int)
    cls_cnt_full[unique_cls] = cls_counts
    
    # Normalize priors
    priors_raw = cls_cnt_full.astype(float)
    priors_min, priors_max = priors_raw.min(), priors_raw.max()
    if priors_max > priors_min:
        priors = 1 - (priors_raw - priors_min) / (priors_max - priors_min)
    else:
        priors = np.ones(class_num)
    
    return priors, cls_cnt_full


def compute_classwise_statistics(preds, targets, keys, class_num):
    """
    Compute class-wise mean and standard deviation statistics.
    
    Args:
        preds: Array of prediction probabilities
        targets: Array of true labels
        keys: List of sample index keys
        class_num: Number of classes
        
    Returns:
        classwise_mean: Mean predictions per class (class_num x class_num)
        classwise_std: Std predictions per class (class_num x class_num)
        zscore_dict_classwise: Dictionary mapping sample keys to z-scores
    """
    classwise_mean = np.zeros((class_num, class_num))
    classwise_std = np.zeros((class_num, class_num))
    zscore_dict_classwise = {}
    classes = np.arange(class_num)
    
    for cls_idx in classes:
        mask = targets == cls_idx
        if mask.any():
            classwise_mean[cls_idx] = preds[mask].mean(axis=0)
            classwise_std[cls_idx] = preds[mask].std(axis=0)
            # Avoid division by zero
            classwise_std[cls_idx] = np.clip(classwise_std[cls_idx], 1e-6, None)
            
            # Compute zscore for all samples in this class (without masking true class)
            zscore_classwise = (preds[mask] - classwise_mean[cls_idx]) / classwise_std[cls_idx]
            
            # Store zscores by sample key
            mask_indices = np.where(mask)[0]
            zscore_dict_classwise.update({keys[i]: zscore_classwise[j] for j, i in enumerate(mask_indices)})
    
    return classwise_mean, classwise_std, zscore_dict_classwise


def zscore(pred_dict, classwise_mean, classwise_std, zscore_dict_classwise, net_id):
    """
    Compute z-scores for all samples with true class masked.
    Note: This function modifies pred_dict in-place by masking the true class probability.
    
    Args:
        pred_dict: Dictionary mapping sample indices to (prob, true_label)
        classwise_mean: Mean predictions per class
        classwise_std: Std predictions per class
        zscore_dict_classwise: Dictionary mapping sample keys to z-scores
        net_id: Network/client identifier
        
    Returns:
        zscores_dict: Dictionary mapping sample keys to z-scores (with true class masked)
        zscore_matrix: Matrix of z-scores for all samples
    """
    zscores_dict = {}
    zscore_matrix = []
    
    for key, (pred_prob, true_cls) in pred_dict.items():
        # Zero out true class probability for z-score calculation

        # IMPORTANT: Modify pred_prob in-place to match original implementation
        # This ensures that when pred_prob is used later (e.g., in nozscoreh mode),
        # it will have the true class already masked
        #pred_prob[true_cls] = 0
        
        # Compute z-score: (pred - mean) / std
        zscore = (pred_prob - classwise_mean[true_cls]) / classwise_std[true_cls]
        zscores_dict[key] = zscore
        zscore_matrix.append(zscore)
        
        # Verification: compare with class-wise computed zscore (before masking true class)
        if key in zscore_dict_classwise:
            zscore_before_mask = zscore_dict_classwise[key].copy()
            #zscore_before_mask[true_cls] = 0  # Mask true class for comparison
            diff = np.abs(zscore - zscore_before_mask)
            max_diff = np.max(diff)

            if max_diff > 1e-5:
                print(f'Client {net_id} - zscore: {zscore}')
                print(f'Client {net_id} - zscore_before_mask: {zscore_before_mask}')
                print(f'Client {net_id} - Warning: zscore mismatch for sample {key}, max diff: {max_diff}')
    
    zscore_matrix = np.array(zscore_matrix)
    return zscores_dict, zscore_matrix


def compute_adaptive_threshold(zscore_matrix, relabel_thre, args, net_id):
    """
    Compute adaptive thresholds for relabeling.
    
    Args:
        zscore_matrix: Matrix of z-scores for all samples
        relabel_thre: Threshold for relabeling (percentage or constant value)
        args: Arguments object
        net_id: Network/client identifier
        
    Returns:
        threclass: Threshold values for each class
    """
    if 'consthre' not in args.id:
        # Adaptive threshold: top thre% samples per class
        # Validate relabel_thre is in valid range [0, 100]
        if not (0 <= relabel_thre <= 100):
            print(f'Client {net_id} - Warning: relabel_thre={relabel_thre} is not in [0, 100], using default 1%')
            relabel_thre = 1.0  # Default to 1% if invalid
        total_samples = zscore_matrix.shape[0]
        m = int(relabel_thre / 100 * total_samples)
        m = max(0, min(m, total_samples - 1))  # Ensure m is in valid range
        k = zscore_matrix.shape[0] - 1 - m  # Convert for ascending partition
        k = max(0, min(k, zscore_matrix.shape[0] - 1))  # Ensure k is in valid range
        
        partitioned = np.partition(zscore_matrix, k, axis=0)
        threclass = partitioned[k, np.arange(zscore_matrix.shape[1])]
        print(f'Client {net_id} - Adaptive thresholds (top {relabel_thre}%): {threclass}')
    else:
        # Constant threshold
        threclass = relabel_thre
        print(f'Client {net_id} - Constant threshold: {relabel_thre}')
    
    return threclass


def get_rho_with_tanh_norm(pred_dict, zscore_dict_classwise, priors, threclass, class_num, args):
    """
    Compute relabel probabilities using tanh normalization and determine relabel classes.
    
    Args:
        pred_dict: Dictionary mapping sample indices to (prob, true_label)
        zscore_dict_classwise: Dictionary mapping sample keys to z-scores
        priors: Normalized prior weights for each class
        threclass: Threshold values for each class
        class_num: Number of classes
        args: Arguments object
        
    Returns:
        relabel_flag: Dictionary mapping sample indices to relabel classes
        label_cnt: Dictionary counting labels per class
        comp: Confusion matrix dictionary
        relabelcnt: Total number of relabeled samples
    """
    relabel_flag = {}
    classes = np.arange(class_num)
    label_cnt = {cls: 0 for cls in classes}
    comp = {cls: {} for cls in classes}
    relabelcnt = 0
    
    for key, (pred_prob, true_cls) in pred_dict.items():
        zscore = zscore_dict_classwise[key]
        
        # Compute prior-weighted flip rate
        prior_weight = np.maximum(priors - priors[true_cls], 0)
        if args.id == 'nozscoreh':
            fliprate = pred_prob * prior_weight
        else:
            fliprate = np.tanh(zscore - threclass) * prior_weight
        
        # Stochastic relabel assignment
        uniform_rand = np.random.rand(class_num)
        relabel_candidates = np.where(fliprate > uniform_rand)[0]
        
        if len(relabel_candidates) > 0:
            # Select class with highest flip rate
            best_idx = relabel_candidates[np.argmax(fliprate[relabel_candidates])]
            relabel_class = classes[best_idx]
            
            relabel_flag[key] = relabel_class
            label_cnt[relabel_class] += 1
            comp[true_cls][relabel_class] = comp[true_cls].get(relabel_class, 0) + 1
            relabelcnt += 1
        else:
            # Keep original label
            label_cnt[true_cls] += 1
    
    return relabel_flag, label_cnt, comp, relabelcnt


def relabel_local_data(g_head, net, net_id, args, relabel_thre, train_dataloader, class_num=100, read=False, store=False, method='loge'):
    """
    Relabel local data based on prediction statistics and z-scores.
    
    Args:
        g_head: Global head classifier
        net: Network backbone
        net_id: Network/client identifier
        args: Arguments object
        relabel_thre: Threshold for relabeling (percentage or constant value)
        train_dataloader: Training data loader
        class_num: Number of classes
        read: If True, load existing relabel info from file
        store: If True, compute and save relabel info to file
        method: Method for computing logits ('loge' or 'etf')
        
    Returns:
        Dictionary with 'relabel_flag' mapping sample indices to predicted relabel classes
    """
    np.set_printoptions(suppress=True, precision=4)
    
    # Early return if neither read nor store
    if not read and not store:
        return None
    
    # Handle read mode
    if read:
        relabel_file = './output/relabels/'+args.id+'_net_'+str(net_id)+'_'+'relabel_info.pkl'
        if os.path.exists(relabel_file):
            with open(relabel_file, 'rb') as j:
                relabel_info = pickle.load(j)
            return relabel_info
        else:
            return None
    
    # ========== Step 1: Collect predictions ==========
    pred_dict, keys, targets, preds = collect_Q(g_head, net, train_dataloader, args, method, num_epochs=5)
    
    # ========== Step 2: Compute normalized priors ==========
    priors, cls_cnt_full = compute_normalized_priors(targets, class_num)
    
    # ========== Step 3: Compute class-wise statistics ==========
    classwise_mean, classwise_std, zscore_dict_classwise = compute_classwise_statistics(preds, targets, keys, class_num)
    
    # ========== Step 4: Compute z-scores for all samples ==========
    zscores_dict, zscore_matrix = zscore(pred_dict, classwise_mean, classwise_std, zscore_dict_classwise, net_id)
    
    # ========== Step 5: Compute adaptive thresholds ==========
    threclass = compute_adaptive_threshold(zscore_matrix, relabel_thre, args, net_id)
    
    # ========== Step 6: Determine relabel classes ==========
    relabel_flag, label_cnt, comp, relabelcnt = get_rho_with_tanh_norm(
        pred_dict, zscore_dict_classwise, priors, threclass, class_num, args
    )
    
    # ========== Step 7: Save results ==========
    relabel_info = {
        'relabel_flag': relabel_flag,
        'relabelcnt': int(relabelcnt),
        'total_samples': int(len(pred_dict)),
        'relabel_ratio': float(relabelcnt / max(len(pred_dict), 1)),
        'label_cnt': label_cnt,
        'confusion_full': comp,
    }
    
    if store:
        os.makedirs('./output/relabels/', exist_ok=True)
        relabel_file = './output/relabels/'+args.id+'_net_'+str(net_id)+'_'+'relabel_info.pkl'
        with open(relabel_file, 'wb') as file:
            pickle.dump(relabel_info, file)
    
    print(f'Client {net_id}: {relabelcnt} samples relabeled. Total samples: {len(pred_dict)}, {relabelcnt/len(pred_dict)*100:.2f}% are relabeled.')
    print(f'Confusion: {comp}, Label counts: {label_cnt}')
    return relabel_info

