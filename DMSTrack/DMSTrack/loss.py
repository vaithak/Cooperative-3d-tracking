import numpy as np
import torch
import math

from AB3DMOT.AB3DMOT_libs.box import Box3D


def get_2d_center_distance_matrix(gt_boxes, track_boxes):
    """
    Calculate the distance matrix using 2D ground plane center distance.
    
    Args:
        gt_boxes (torch.Tensor): Ground truth boxes [M, 7], format [x,y,z,theta,l,w,h]
        track_boxes (torch.Tensor): Track boxes [N, 7], format [x,y,z,theta,l,w,h]
        
    Returns:
        torch.Tensor: Distance matrix [M, N]
    
    Note:
        Uses AB3DMOT KITTI coordinate system where xz is the ground plane, y is height
    """
    # Extract x,z coordinates for distance calculation (ground plane)
    gt_centers = torch.cat([gt_boxes[:, 0:1], gt_boxes[:, 2:3]], dim=2).unsqueeze(1)  # [M, 1, 2]
    track_centers = torch.cat([track_boxes[:, 0:1], track_boxes[:, 2:3]], dim=2).unsqueeze(0)  # [1, N, 2]
    
    # Calculate Euclidean distance matrix
    distance_matrix = torch.norm(gt_centers - track_centers, dim=2)  # [M, N]
    
    return distance_matrix


def get_neg_log_likelihood_loss(mean_boxes, covariance, gt_boxes):
    """
    Calculate negative log likelihood loss between predicted boxes and ground truth.
    
    Args:
        mean_boxes (torch.Tensor): Mean predicted boxes [N, 7], format [x,y,z,theta,l,w,h]
        covariance (torch.Tensor): Covariance matrices [N, 7, 7]
        gt_boxes (torch.Tensor): Ground truth boxes [G, 7], format [x,y,z,theta,l,w,h]
        
    Returns:
        tuple: (neg_log_likelihood, matched_det_count)
            - neg_log_likelihood (torch.Tensor): Negative log likelihood for each detection [N]
            - matched_det_count (int): Number of detections matched to ground truths
    """
    N, V = mean_boxes.shape
    
    # Get diagonal elements from covariance matrices
    covariance_diag = torch.diagonal(covariance, dim1=-2, dim2=-1)  # [N, 7]
    
    # Find closest ground truth for each detection
    distance_matrix = get_2d_center_distance_matrix(gt_boxes, mean_boxes)  # [G, N]
    min_per_det = torch.min(distance_matrix, dim=0)
    min_distance_per_det = min_per_det.values
    matched_gt_idx_per_det = min_per_det.indices
    
    # Get matched ground truths
    matched_gt_per_det = gt_boxes[matched_gt_idx_per_det]  # [N, 7]
    
    # Only consider detections whose distance to closest GT < 2 meters
    DISTANCE_THRESHOLD = 2.0
    matched_mask_per_det = min_distance_per_det < DISTANCE_THRESHOLD
    matched_det_count = torch.sum(matched_mask_per_det)
    
    # Calculate difference between predicted boxes and matched ground truths
    diff = mean_boxes - matched_gt_per_det  # [N, V]
    
    # Calculate norm of diagonal covariance for normalization
    covariance_diag_norm = torch.norm(covariance_diag, dim=1)  # [N]
    
    # Calculate negative log likelihood (assuming diagonal covariance)
    neg_log_likelihood = (
        0.5 * V * math.log(2 * math.pi) +
        0.5 * torch.log(covariance_diag_norm) +
        0.5 * torch.sum(diff ** 2 / covariance_diag, dim=1)
    )  # [N]
    
    # Apply mask to only include detections matched to ground truths
    neg_log_likelihood = neg_log_likelihood * matched_mask_per_det
    
    return neg_log_likelihood, matched_det_count


def transform_dets_dict_to_kf_format(dets_dict, dtype, device):
    """
    Transform detection dictionary to format compatible with Kalman filter.
    
    Args:
        dets_dict (dict): Dictionary mapping CAV IDs to Box3D objects
        dtype (torch.dtype): Desired data type
        device (torch.device): Device to place tensor on
        
    Returns:
        torch.Tensor: Detections in format [num_total_dets, 7], where 7=[x,y,z,theta,l,w,h]
    """
    dets = []
    for boxes in dets_dict.values():
        for box3d in boxes:
            dets.append(Box3D.bbox2array(box3d))
    
    if not dets:
        return torch.zeros((0, 7), dtype=dtype, device=device)
        
    dets = np.stack(dets, axis=0)
    dets = torch.tensor(dets, dtype=dtype, device=device)
    return dets


def get_matched_gt_id_for_each_det(dets, gt_boxes, gt_ids, dataset, match_gt_threshold=1):
    """
    For each detection, get the matched ground truth IDs.
    
    Args:
        dets (torch.Tensor): Detections [M, 7], format [x,y,z,theta,l,w,h]
        gt_boxes (torch.Tensor): Ground truth boxes [N, 7], format [x,y,z,theta,l,w,h]
        gt_ids (list): Ground truth IDs [N]
        dataset (str): Dataset name ('nuscenes', 'kitti', or 'v2v4real')
        match_gt_threshold (float): Distance threshold for matching
        
    Returns:
        tuple: (closest_gt_ids, matched_mask)
            - closest_gt_ids (list): ID of closest GT for each detection
            - matched_mask (torch.Tensor): Boolean mask of detections matched to GTs
    """
    if dets.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return [], torch.zeros(0, dtype=torch.bool, device=dets.device)
    
    # Get appropriate coordinates based on dataset
    if dataset == 'nuscenes':
        # For nuScenes, we use xy plane
        dets_centers = dets[:, :2].unsqueeze(1)  # [M, 1, 2]
        gts_centers = gt_boxes[:, :2].unsqueeze(0)  # [1, N, 2]
    else:
        # For KITTI or v2v4real, we use xz plane
        dets_centers = torch.cat([dets[:, 0:1], dets[:, 2:3]], dim=1).unsqueeze(1)  # [M, 1, 2]
        gts_centers = torch.cat([gt_boxes[:, 0:1], gt_boxes[:, 2:3]], dim=1).unsqueeze(0)  # [1, N, 2]
    
    # Calculate distance matrix
    diff = dets_centers - gts_centers  # [M, N, 2]
    distance_matrix = torch.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)
    
    # Find closest GT for each detection
    closest_info = torch.min(distance_matrix, dim=1)
    closest_gts_indices = closest_info.indices
    closest_gt_ids = [gt_ids[idx] for idx in closest_gts_indices]
    closest_dists = closest_info.values
    
    # Create mask for detections within threshold
    matched_mask = closest_dists <= match_gt_threshold
    
    return closest_gt_ids, matched_mask


def get_samples_masks(negative_sample_mode, closest_gts_per_det, matched_mask_per_det, 
                     closest_gts_per_trk, matched_mask_per_trk):
    """
    Generate masks for positive and negative sample pairs.
    
    Args:
        negative_sample_mode (int): 0 or 1, controls how negative samples are selected
        closest_gts_per_det (list): List of closest GT IDs for each detection
        matched_mask_per_det (torch.Tensor): Mask of matched detections [num_dets]
        closest_gts_per_trk (list): List of closest GT IDs for each track
        matched_mask_per_trk (torch.Tensor): Mask of matched tracks [num_tracks]
        
    Returns:
        tuple: (positive_samples_mask, negative_samples_mask)
            - positive_samples_mask (torch.Tensor): Mask for positive samples [num_dets, num_tracks]
            - negative_samples_mask (torch.Tensor): Mask for negative samples [num_dets, num_tracks]
    """
    dtype = matched_mask_per_det.dtype
    device = matched_mask_per_det.device
    
    # Valid pairs: both detection and track match to a ground truth
    valid_mask = matched_mask_per_det.unsqueeze(1) & matched_mask_per_trk.unsqueeze(0)
    
    # Create mask for pairs matching to the same GT ID
    have_same_closest_gt_id_mask = torch.zeros_like(valid_mask)
    
    for d in range(len(closest_gts_per_det)):
        for t in range(len(closest_gts_per_trk)):
            have_same_closest_gt_id_mask[d][t] = (closest_gts_per_det[d] == closest_gts_per_trk[t])
    
    # Positive samples: valid pairs matching to the same GT
    positive_samples_mask = valid_mask & have_same_closest_gt_id_mask
    
    # Negative samples based on mode
    if negative_sample_mode == 0:
        # Only valid pairs matching to different GTs
        negative_samples_mask = valid_mask & ~have_same_closest_gt_id_mask
    elif negative_sample_mode == 1:
        # All pairs that are not positive
        negative_samples_mask = ~positive_samples_mask
    else:
        raise ValueError(f"Invalid negative_sample_mode: {negative_sample_mode}")
    
    return positive_samples_mask, negative_samples_mask


def get_loss_from_samples(dtype, device, distance_matrix, positive_samples_mask, negative_samples_mask, 
                         contrastive_margin, match_dist_threshold, positive_sample_loss_weight):
    """
    Calculate contrastive and margin losses from positive and negative samples.
    
    Args:
        dtype (torch.dtype): Data type for tensors
        device (torch.device): Device for tensors
        distance_matrix (torch.Tensor): Distance matrix between detections and tracks [num_dets, num_tracks]
        positive_samples_mask (torch.Tensor): Mask for positive samples [num_dets, num_tracks]
        negative_samples_mask (torch.Tensor): Mask for negative samples [num_dets, num_tracks]
        contrastive_margin (float): Margin for contrastive loss
        match_dist_threshold (float): Distance threshold for matching
        positive_sample_loss_weight (float): Weight for positive samples in the loss
        
    Returns:
        tuple: (single_margin_loss, contrastive_loss, num_samples)
            - single_margin_loss (torch.Tensor): Margin loss for individual samples
            - contrastive_loss (torch.Tensor): Contrastive loss
            - num_samples (int): Number of sample pairs used in loss calculation
    """
    num_positive_samples = torch.sum(positive_samples_mask)
    num_negative_samples = torch.sum(negative_samples_mask)
    
    # If no positive or negative samples, return zero loss
    if num_positive_samples == 0 or num_negative_samples == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), torch.tensor(0.0, dtype=dtype, device=device), 0
    
    # Ensure distance matrix is valid
    assert not torch.any(torch.isnan(distance_matrix))
    
    # Extract distances for positive and negative samples
    positive_distances = distance_matrix[positive_samples_mask]
    negative_distances = distance_matrix[negative_samples_mask]
    
    # Calculate contrastive loss (we want positive_distances < negative_distances)
    diff = negative_distances.unsqueeze(0) - positive_distances.unsqueeze(1)  # [P, N]
    contrastive_max_margin_diff = torch.max(torch.zeros_like(diff), contrastive_margin - diff)
    contrastive_loss = torch.mean(contrastive_max_margin_diff)
    
    # Calculate margin loss for individual samples
    # We want positive pairs to have distance < (match_dist_threshold - margin/2)
    # and negative pairs to have distance > (match_dist_threshold + margin/2)
    margin_half = contrastive_margin / 2
    
    positive_max_margin_diff = torch.max(
        torch.zeros_like(positive_distances), 
        margin_half - (match_dist_threshold - positive_distances)
    )
    positive_max_margin_loss = torch.mean(positive_max_margin_diff)
    
    negative_max_margin_diff = torch.max(
        torch.zeros_like(negative_distances), 
        margin_half - (negative_distances - match_dist_threshold)
    )
    negative_max_margin_loss = torch.mean(negative_max_margin_diff)
    
    # Combine losses with weighting
    sample_count = num_positive_samples * num_negative_samples
    weighted_pos_loss = positive_max_margin_loss * sample_count * positive_sample_loss_weight
    weighted_neg_loss = negative_max_margin_loss * sample_count * (1 - positive_sample_loss_weight)
    single_margin_loss = (weighted_pos_loss + weighted_neg_loss) / sample_count
    
    return single_margin_loss, contrastive_loss, sample_count


def get_association_loss(dets_dict, trackers, gt_boxes, gt_ids, prev_gt_boxes, prev_gt_ids):
    """
    Calculate association loss between detections and tracks.
    
    Args:
        dets_dict (dict): Dictionary mapping CAV IDs to Box3D detection objects
        trackers (list): List of DKF tracking objects
        gt_boxes (torch.Tensor): Current frame ground truth boxes [num_gts, 7]
        gt_ids (list): Current frame ground truth IDs [num_gts]
        prev_gt_boxes (torch.Tensor): Previous frame ground truth boxes [prev_num_gts, 7]
        prev_gt_ids (list): Previous frame ground truth IDs [prev_num_gts]
        
    Returns:
        tuple: (association_loss, association_loss_count)
            - association_loss (torch.Tensor): Total association loss
            - association_loss_count (int): Number of sample pairs used in loss calculation
    """
    dtype = gt_boxes.dtype
    device = gt_boxes.device
    
    # No loss for first frame in a sequence
    if prev_gt_boxes is None:
        return torch.tensor(0.0, dtype=dtype, device=device), 0
    
    # No ground truth boxes
    if len(gt_boxes) == 0 or len(prev_gt_boxes) == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), 0
    
    # Get previous track states (before prediction)
    prev_trks = torch.stack([trk.dkf.prev_x[:7, 0] for trk in trackers])
    
    # Transform detections to the same format
    dets = transform_dets_dict_to_kf_format(dets_dict, gt_boxes.dtype, gt_boxes.device)
    
    # No detections or tracks
    if dets.shape[0] == 0 or prev_trks.shape[0] == 0:
        return torch.tensor(0.0, dtype=dtype, device=device), 0
    
    # Default dataset for coordinate system
    dataset = 'v2v4real'
    
    # Get GT matches for detections and tracks
    closest_gts_per_det, matched_mask_per_det = get_matched_gt_id_for_each_det(
        dets, gt_boxes, gt_ids, dataset)
    closest_gts_per_trk, matched_mask_per_trk = get_matched_gt_id_for_each_det(
        prev_trks, prev_gt_boxes, prev_gt_ids, dataset)
    
    # Use negative sample mode 1 (all non-positive pairs are negative)
    negative_sample_mode = 1
    positive_samples_mask, negative_samples_mask = get_samples_masks(
        negative_sample_mode, 
        closest_gts_per_det, matched_mask_per_det, 
        closest_gts_per_trk, matched_mask_per_trk
    )
    
    # Get current track states (after prediction)
    # Need .clone() to avoid inplace operation errors during backprop
    trks = torch.stack([trk.dkf.x[:7, 0].clone() for trk in trackers])
    assert prev_trks.shape == trks.shape
    
    # Loss hyperparameters
    match_dist_threshold = 2.0  # Positive pairs have dist < 2, negative pairs have dist > 2
    contrastive_margin = 2.0    # We want (neg_dist - pos_dist) > 2
    positive_sample_loss_weight = 0.5  # Equal weighting between positive and negative samples
    
    # Calculate distance matrix and losses
    distance_matrix = get_2d_center_distance_matrix(dets, trks)
    
    single_margin_loss, contrastive_loss, loss_count = get_loss_from_samples(
        dtype, device, distance_matrix, positive_samples_mask, negative_samples_mask,
        contrastive_margin, match_dist_threshold, positive_sample_loss_weight
    )
    
    # Combine losses
    association_loss = (single_margin_loss + contrastive_loss) * loss_count
    
    return association_loss, loss_count