import numpy as np
import torch
import torch.nn as nn
import copy
import math
from AB3DMOT.AB3DMOT_libs.model import AB3DMOT
from AB3DMOT.AB3DMOT_libs.box import Box3D
from AB3DMOT.AB3DMOT_libs.matching import data_association

from differentiable_kalman_filter import DKF
from loss import get_2d_center_distance_matrix, get_association_loss, get_neg_log_likelihood_loss


class ObservationCovarianceNet(nn.Module):
    """Neural network for predicting observation covariance matrices"""
    def __init__(self, differentiable_kalman_filter_config, feature='fusion'):
        super(ObservationCovarianceNet, self).__init__()
        self.dim_x = differentiable_kalman_filter_config['dim_x']  # state dimension (10)
        self.dim_z = differentiable_kalman_filter_config['dim_z']  # observation dimension (7)
        self.dkf_type = differentiable_kalman_filter_config['dkf_type']  # single_sensor or multi_sensor
        self.feature = feature
        
        # Feature sizes
        observation_covariance_setting = differentiable_kalman_filter_config['observation_covariance_setting']
        self.bev_feature_channel_size = observation_covariance_setting['feature_channel_size']
        self.bev_feature_region_size = observation_covariance_setting['feature_region_size']
        self.positional_embedding_size = 18 * 256
        
        # Calculate fusion channel size based on sensor type
        if self.dkf_type == 'multi_sensor':
            self.fusion_channel_size = self.bev_feature_channel_size * 4 * 4 + self.positional_embedding_size
        else:  # single_sensor
            self.fusion_channel_size = self.bev_feature_channel_size * 3 * 3 + self.positional_embedding_size
        
        # BEV feature processing layers
        self.bev_conv_and_max_pool = self._create_bev_conv_layers()
        self.bev_linear = nn.Sequential(
            nn.Linear(self.bev_feature_channel_size * 4 * 4, self.bev_feature_channel_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.bev_feature_channel_size, self.dim_x)
        )
        
        # Positional encoding processing layers
        self.positional_encoding_linear = self._create_positional_encoding_layers()
        
        # Fusion layers
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.fusion_channel_size, self.fusion_channel_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_channel_size // 2, self.dim_x),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )
        
        # Create sensor-specific convolutional layers
        self.fusion_bev_conv_and_max_pool = self._create_fusion_bev_conv_layers()
        
        # Fusion positional encoding layers
        self.fusion_positional_encoding_linear = nn.Sequential(
            nn.Linear(self.positional_embedding_size, self.positional_embedding_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.positional_embedding_size, self.positional_embedding_size),
            nn.ReLU(inplace=True),
        )

    def _create_bev_conv_layers(self):
        """Create BEV convolutional layers"""
        return nn.Sequential(
            nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def _create_positional_encoding_layers(self):
        """Create positional encoding layers"""
        return nn.Sequential(
            nn.Linear(self.positional_embedding_size, self.positional_embedding_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.positional_embedding_size // 2, self.positional_embedding_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.positional_embedding_size // 4, self.positional_embedding_size // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.positional_embedding_size // 8, self.positional_embedding_size // 16),
            nn.ReLU(inplace=True),
            nn.Linear(self.positional_embedding_size // 16, self.dim_x)
        )

    def _create_fusion_bev_conv_layers(self):
        """Create fusion BEV convolutional layers based on sensor type"""
        if self.dkf_type == 'multi_sensor':
            return nn.Sequential(
                nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.AvgPool2d(kernel_size=2),
            )
        else:  # single_sensor
            return nn.Sequential(
                nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.bev_feature_channel_size, self.bev_feature_channel_size, kernel_size=3, padding=0, bias=True),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, frame, transformation_matrix, dets_in_gt_order):
        """
        Process detection features to predict covariance matrices
        
        Args:
            x: Detection features [N, C, H, W]
            frame: Frame number
            transformation_matrix: [4, 4] CAV sensor to ego coordinate transformation 
            dets_in_gt_order: [N, 7] detection boxes in format [x, y, z, theta, l, w, h]
            
        Returns:
            Covariance diagonal residual [N, dim_x=10]
        """
        N, C, W, H = x.shape
        if N == 0:
            return torch.zeros([0, self.dim_x], dtype=x.dtype, device=x.device)

        return self.forward_fusion(x, frame, transformation_matrix, dets_in_gt_order)

    def forward_fusion(self, x, frame, transformation_matrix, dets_in_gt_order):
        """Process features using fusion of positional embedding and BEV features"""
        # Process positional embedding
        positional_embedding, _, _, _ = self.get_positional_embedding(
            transformation_matrix, dets_in_gt_order)
        positional_embedding = torch.flatten(positional_embedding, start_dim=1)
        positional_embedding = self.fusion_positional_encoding_linear(positional_embedding)
        
        # Process BEV features
        N, C, H, W = x.shape
        bev_feature = self.fusion_bev_conv_and_max_pool(x)
        bev_feature = bev_feature.reshape([N, -1])
        
        # Fusion and final processing
        fusion_feature = torch.cat([positional_embedding, bev_feature], dim=1)
        output = self.fusion_linear(fusion_feature)
        
        return output
    
    def get_positional_embedding(self, transformation_matrix, dets_in_gt_order, hidden_dim=256):
        """
        Create positional embeddings for detections relative to ego vehicle and sensors
        
        Args:
            transformation_matrix: [4, 4] CAV sensor to ego coordinate transformation
            dets_in_gt_order: [N, 7] detection boxes in format [x, y, z, theta, l, w, h]
            hidden_dim: Dimension of embedding features
            
        Returns:
            positional_embedding: [N, 18, hidden_dim] embedding tensor
            distance_2d_det_to_ego: Distance from detection to ego
            distance_2d_det_to_sensor: Distance from detection to sensor
            distance_2d_sensor_to_ego: Distance from sensor to ego
        """
        N, dim_z = dets_in_gt_order.shape

        # Calculate distances and relative positions
        det_to_ego_x = dets_in_gt_order[:, 0]
        det_to_ego_y = dets_in_gt_order[:, 1]
        det_to_ego_z = dets_in_gt_order[:, 2]
        distance_2d_det_to_ego = torch.sqrt(det_to_ego_x ** 2 + det_to_ego_z ** 2)

        # Extract sensor position from transformation matrix
        sensor_to_ego_x = transformation_matrix[0, 3]
        sensor_to_ego_y = transformation_matrix[2, 3]
        sensor_to_ego_z = transformation_matrix[1, 3]
        distance_2d_sensor_to_ego = torch.sqrt(sensor_to_ego_x ** 2 + sensor_to_ego_z ** 2)
        sensor_to_ego_theta = torch.arctan2(transformation_matrix[1,0], transformation_matrix[0,0])

        # Calculate detection position relative to sensor
        det_to_sensor_x = det_to_ego_x - sensor_to_ego_x
        det_to_sensor_y = det_to_ego_y - sensor_to_ego_y
        det_to_sensor_z = det_to_ego_z - sensor_to_ego_z
        distance_2d_det_to_sensor = torch.sqrt((det_to_sensor_x) ** 2 + (det_to_sensor_z) ** 2)
        det_to_sensor_theta = dets_in_gt_order[:, 3] - sensor_to_ego_theta

        # Combine all positional features
        positional_feature = torch.cat(
            [
                dets_in_gt_order,
                distance_2d_det_to_ego.unsqueeze(1),
                sensor_to_ego_x.unsqueeze(0).unsqueeze(0).expand(N, 1),
                sensor_to_ego_y.unsqueeze(0).unsqueeze(0).expand(N, 1),
                sensor_to_ego_z.unsqueeze(0).unsqueeze(0).expand(N, 1),
                sensor_to_ego_theta.unsqueeze(0).unsqueeze(0).expand(N, 1),
                distance_2d_sensor_to_ego.unsqueeze(0).unsqueeze(0).expand(N, 1),
                det_to_sensor_x.unsqueeze(1),
                det_to_sensor_y.unsqueeze(1),
                det_to_sensor_z.unsqueeze(1),
                det_to_sensor_theta.unsqueeze(1),
                distance_2d_det_to_sensor.unsqueeze(1)
            ],
            dim=1
        )

        # Normalize all distances by max distance (200 meters)
        max_distance = 200
        positional_feature /= max_distance
        # Revert normalization for angles
        positional_feature[:, 3] *= max_distance
        positional_feature[:, 11] *= max_distance
        positional_feature[:, 16] *= max_distance

        # Create sinusoidal positional embeddings
        half_hidden_dim = hidden_dim // 2
        scale = math.pi
        dim_t = torch.arange(half_hidden_dim, dtype=positional_feature.dtype, device=positional_feature.device)
        dim_t = 2 ** (2 * dim_t / hidden_dim)
        
        positional_embedding = positional_feature.unsqueeze(dim=2)
        positional_embedding = positional_embedding * scale / dim_t
        positional_embedding = torch.cat([positional_embedding.sin(), positional_embedding.cos()], dim=2)

        return positional_embedding, distance_2d_det_to_ego, distance_2d_det_to_sensor, distance_2d_sensor_to_ego


class DMSTrack(AB3DMOT):
    """Differentiable Probabilistic 3D Multi-Object Collaborative Tracker"""
    def __init__(self, cfg, cat, calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=None, ID_init=0, 
                dtype=None, device=None, differentiable_kalman_filter_config=None, 
                observation_covariance_net_dict=None, force_gt_as_predicted_track=False, 
                use_static_default_R=False, use_multiple_nets=False):
        super().__init__(cfg, cat, calib, oxts, img_dir, vis_dir, hw, log)
        
        # Initialize parameters
        self.dtype = dtype
        self.device = device
        self.dim_x = differentiable_kalman_filter_config['dim_x']
        self.dim_z = differentiable_kalman_filter_config['dim_z']
        self.observation_covariance_net_dict = observation_covariance_net_dict
        self.gt_data_association_threshold = differentiable_kalman_filter_config['gt_data_association_threshold']
        
        # Previous frames' ground-truth for loss calculation
        self.prev_gt_boxes = None
        self.prev_gt_ids = None
        self.prev_prev_gt_boxes = None
        self.prev_prev_gt_ids = None
        
        # Configuration flags
        self.force_gt_as_predicted_track = force_gt_as_predicted_track
        self.use_static_default_R = use_static_default_R
        self.use_multiple_nets = use_multiple_nets

    def get_param(self, cfg, cat):
        """Get tracking algorithm parameters"""
        if cfg.dataset == 'v2v4real':
            # Use default AB3DMOT KITTI car settings
            algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
        else:
            assert False, "Unsupported dataset"

        # Adjust threshold sign if needed
        if metric in ['dist_3d', 'dist_2d', 'm_dis']: 
            thres *= -1
            
        self.algm, self.metric, self.thres, self.max_age, self.min_hits = algm, metric, thres, max_age, min_hits

        # Define similarity score ranges
        if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: 
            self.max_sim, self.min_sim = 0.0, -100.
        elif self.metric in ['iou_2d', 'iou_3d']:          
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ['giou_2d', 'giou_3d']:        
            self.max_sim, self.min_sim = 1.0, -1.0

    def within_range_torch(self, angle):
        """Normalize angle to [-pi, pi] range"""
        if angle > math.pi:
            angle -= 2 * math.pi
        if angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def transform_gt_as_predicted_track(self, frame):
        """Use ground truth from previous frames to generate tracks for current frame"""
        self.trackers = []
        trks = []

        for i in range(self.prev_gt_boxes.shape[0]):
            prev_box = self.prev_gt_boxes[i]
            gt_id = self.prev_gt_ids[i]
            i_in_prev_prev_frame = (self.prev_prev_gt_ids == gt_id).nonzero(as_tuple=True)[0]

            bbox3D = prev_box.clone()

            # Apply velocity if object was detected in frame before previous
            if len(i_in_prev_prev_frame) != 0:
                prev_prev_box = self.prev_prev_gt_boxes[i_in_prev_prev_frame][0]
                bbox3D[:2] += prev_box[:2] - prev_prev_box[:2]
        
            info = np.array([0, 2, 0, 0, 0, 0, 1.0])
            # Create DKF tracker
            trk = DKF(bbox3D.detach().cpu().numpy(), info, gt_id.item(), self.dtype, self.device, 
                self.use_static_default_R, frame, 'ego', gt_id.item())

            # Set up tracker
            trk.hits = 3  # Force it to show up in tracking results
            trk.dkf.prev_x = torch.cat([prev_box.reshape([7, 1]), 
                                       torch.zeros([3, 1], dtype=self.dtype, device=self.device)], dim=0)

            self.trackers.append(trk)

            # Add track in Box3D format
            trk_tmp = trk.dkf.x.reshape((-1))[:7].detach().cpu().numpy()
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def prediction(self):
        """
        Perform Kalman filter prediction for all trackers
        Returns predicted track states in numpy format for data association
        """
        trks = []
        for t in range(len(self.trackers)):
            # Get tracker and perform prediction
            kf_tmp = self.trackers[t]
            if kf_tmp.id == self.debug_id:
                print('\n before prediction')
                print(kf_tmp.dkf.x.reshape((-1)))
                print('\n current velocity')
                print(kf_tmp.get_velocity())
            
            kf_tmp.dkf.predict(None)
            
            if kf_tmp.id == self.debug_id:
                print('After prediction')
                print(kf_tmp.dkf.x.reshape((-1)))
                
            # Normalize orientation
            kf_tmp.dkf.x[3] = self.within_range_torch(kf_tmp.dkf.x[3])

            # Reset matched detection info for visualization
            kf_tmp.reset_matched_detection_id_dict()

            # Update tracking statistics
            kf_tmp.time_since_update += 1
            
            # Extract track state and convert to Box3D format
            trk_tmp = kf_tmp.dkf.x.reshape((-1))[:7].detach().cpu().numpy()
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def get_trks_for_match(self):
        """
        Get current tracker states in Box3D format without performing prediction
        Used when iterating through multiple sensors' detections
        """
        trks = []
        for t in range(len(self.trackers)):
            kf_tmp = self.trackers[t]
            
            # Debug printing if needed
            if kf_tmp.id == self.debug_id:
                print('\n before prediction')
                print(kf_tmp.dkf.x.reshape((-1)))
                print('\n current velocity')
                print(kf_tmp.get_velocity())
                print('After prediction')
                print(kf_tmp.dkf.x.reshape((-1)))
                
            # Normalize orientation
            kf_tmp.dkf.x[3] = self.within_range_torch(kf_tmp.dkf.x[3])

            # Extract track state and convert to Box3D format
            trk_tmp = kf_tmp.dkf.x.reshape((-1))[:7].detach().cpu().numpy()
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def orientation_correction_torch(self, theta_pre, theta_obs):
        """
        Correct orientation angles between track prediction and observation
        Ensures angles are comparable (within 90 degrees of each other)
        """
        # Normalize angles to [-pi, pi]
        theta_pre = self.within_range_torch(theta_pre)
        theta_obs = self.within_range_torch(theta_obs)

        # If angles differ by more than 90° but less than 270°, flip one by 180°
        if abs(theta_obs - theta_pre) > math.pi / 2.0 and abs(theta_obs - theta_pre) < math.pi * 3 / 2.0:
            theta_pre += math.pi
            theta_pre = self.within_range_torch(theta_pre)

        # Handle edge case when angles differ by more than 270°
        if abs(theta_obs - theta_pre) >= math.pi * 3 / 2.0:
            if theta_obs > 0: 
                theta_pre += math.pi * 2
            else: 
                theta_pre -= math.pi * 2

        return theta_pre, theta_obs

    def get_learnable_observation_covariance(self, default_init_P, diagonal_residual):
        """
        Create learnable observation covariance matrices based on network outputs
        
        Args:
            default_init_P: Default initial state covariance matrix [10, 10]
            diagonal_residual: Network-predicted adjustments [N, 10]
            
        Returns:
            learnable_init_P: Learnable initial state covariance [N, 10, 10]
            learnable_R: Learnable observation covariance [N, 7, 7]
        """
        # Calculate diagonal elements with residual
        diagonal_P = (torch.sqrt(torch.diag(default_init_P)) + diagonal_residual) ** 2
        
        # Prevent too small values that could cause numerical issues
        diagonal_P = torch.clamp(diagonal_P, min=1e-2)
        
        # Create diagonal matrices for each detection
        learnable_init_P = []
        for i in range(diagonal_P.shape[0]):
            single_learnable_init_P = torch.diag(diagonal_P[i])
            learnable_init_P.append(single_learnable_init_P)

        if len(learnable_init_P) == 0:
            # Handle case with no detections
            return torch.zeros([0, 10], dtype=self.dtype, device=self.device), \
                   torch.zeros([0, 7], dtype=self.dtype, device=self.device)

        # Stack matrices
        learnable_init_P = torch.stack(learnable_init_P, dim=0)
        
        # Extract observation portion (first 7x7 block)
        learnable_R = learnable_init_P[:, :7, :7]

        return learnable_init_P, learnable_R

    def update(self, matched, unmatched_trks, dets, info, learnable_R_dict, frame, cav_id):
        """
        Update matched trackers with assigned detections
        
        Args:
            matched: Indices of matched detections and tracks
            unmatched_trks: Indices of unmatched tracks
            dets: List of detections
            info: Additional info for each detection
            learnable_R_dict: Dictionary of observation covariance matrices
            frame: Current frame number
            cav_id: ID of the connected autonomous vehicle
        """
        assert(len(dets) == learnable_R_dict[cav_id].shape[0])
        dets = copy.copy(dets)
        
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                # Find the matched detection for this track
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                assert len(d) == 1, 'Each track should match exactly one detection'
                
                # Update statistics
                trk.time_since_update = 0
                trk.hits += 1

                # Correct orientation between track and detection
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.dkf.x[3], bbox3d[3] = self.orientation_correction_torch(trk.dkf.x[3], bbox3d[3])

                # Get observation covariance for this detection
                learnable_R = learnable_R_dict[cav_id][d[0]]

                # Check if this track was already updated in this frame by another sensor
                skip_update = trk.last_updated_frame == frame
                skip_update = False  # Override to always update

                if not skip_update:
                    # Update track with detection
                    trk.dkf.update(bbox3d, learnable_R, None)
                    trk.last_updated_frame = frame
                    trk.last_updated_cav_id = cav_id
                    trk.matched_detection_id_dict[cav_id] = d[0]

                # Debug printing
                if trk.id == self.debug_id:
                    print('after matching')
                    print(trk.dkf.x.reshape((-1)))
                    print('\n current velocity')
                    print(trk.get_velocity())

                # Normalize orientation
                trk.dkf.x[3] = self.within_range_torch(trk.dkf.x[3])
                trk.info = info[d, :][0]

    def birth(self, dets, info, unmatched_dets, frame, cav_id, learnable_init_P_dict):
        """
        Create and initialize new trackers for unmatched detections
        
        Args:
            dets: List of detections
            info: Additional info for each detection
            unmatched_dets: Indices of unmatched detections
            frame: Current frame number
            cav_id: ID of the connected autonomous vehicle
            learnable_init_P_dict: Dictionary of learnable initial covariance matrices
            
        Returns:
            List of new track IDs
        """
        new_id_list = []
        for i in unmatched_dets:
            # Create new tracker
            trk = DKF(
                Box3D.bbox2array(dets[i]), 
                info[i, :], self.ID_count[0], 
                self.dtype, self.device, self.use_static_default_R,
                frame, cav_id, i, 
                learnable_init_P_dict[cav_id][i]
            )
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            self.ID_count[0] += 1

        return new_id_list

    def output(self):
        """
        Output existing tracks that have been stably associated
        Delete tracks that have not been updated for too long
        
        Returns:
            results: List of tracking results
            matched_detection_id_dict: Dictionary of matched detection IDs
            track_P: List of track covariance matrices
        """
        num_trks = len(self.trackers)
        results = []
        matched_detection_id_dict = []
        track_P = []

        for trk in reversed(self.trackers):
            # Convert track state to output format
            trk_tmp = trk.dkf.x.reshape((-1))[:7].detach().cpu().numpy()
            d = Box3D.array2bbox(trk_tmp)
            d = Box3D.bbox2array_raw(d)

            # Only output tracks that are stable and recent
            if ((trk.time_since_update < self.max_age) and 
                (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
                # Combine state with ID and info
                results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1))
                
                # For visualization and debugging
                matched_detection_id_dict.append(trk.matched_detection_id_dict)
                track_P.append(trk.dkf.P.detach().cpu().numpy())

            num_trks -= 1

            # Remove tracks that haven't been updated for too long
            if trk.time_since_update >= self.max_age:
                self.trackers.pop(num_trks)

        return results, matched_detection_id_dict, track_P


    def greedy_match(self, distance_matrix):
      '''
      Find the one-to-one matching using greedy allgorithm choosing small distance
      distance_matrix: (num_detections, num_tracks)
      '''
      matched_indices = []
      matched_mask = np.zeros_like(distance_matrix)

      num_detections, num_tracks = distance_matrix.shape
      distance_1d = distance_matrix.reshape(-1)
      index_1d = np.argsort(distance_1d)
      index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
      detection_id_matches_to_tracking_id = [-1] * num_detections
      tracking_id_matches_to_detection_id = [-1] * num_tracks
      for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
          tracking_id_matches_to_detection_id[tracking_id] = detection_id
          detection_id_matches_to_tracking_id[detection_id] = tracking_id
          matched_indices.append([detection_id, tracking_id])
          matched_mask[detection_id][tracking_id] = 1

      matched_indices = np.array(matched_indices)
      return matched_indices, matched_mask


    def get_regression_loss_per_pair(self, gt_box, tracking_box):
      '''
      Calculate the regression loss by
      between a ground truth box and a tracking result box
      Input:
        gt_box: ground truth box: [7=self.dim_z]
        tracking_box: tracking result box: [7=self.dim_z]
          7: [x, y, z, theta, l, w, h]
      Output:
        regression_loss_per_pair: a scalar tensor
      '''
      #print('gt_box: ', gt_box)
      #print('tracking_box: ', tracking_box)
      gt_box[3], tracking_box[3] = self.orientation_correction_torch(gt_box[3], tracking_box[3])
      regression_loss_per_pair = torch.norm(gt_box - tracking_box)

      return regression_loss_per_pair


    def get_regression_loss(self, gt_boxes):
      '''
      Calculate the regression loss by
      data association using center point distance 
      between tracking result boxes and ground truth boxes
      Input:
        gt_boxes: ground truth boxes: [N, 7=self.dim_z]
          7: [x, y, z, theta, l, w, h]
        self.trackers: list of class DKF
          DFK.dkf: class DifferentiableKalmanFilter
            DFK.dfk.x: tracking object state: [10]
              10: [x, y, z, theta, l, w, h, dx, dy, dz]
      Output:
        regression_loss_sum: sum of l2 loss of matched track gt pairs
        regression_loss_count: number of matched track gt pairs
      '''
      tracking_boxes = []
      for m in range(len(self.trackers)):
        tracking_boxes.append(self.trackers[m].dkf.x[:self.dim_z, 0])
      tracking_boxes = torch.stack(tracking_boxes, dim=0)

      # this data association does not need grad
      # we only use the matching indices
      with torch.no_grad():
        distance_matrix = get_2d_center_distance_matrix(gt_boxes, tracking_boxes)
        #print('distance_matrix: ', distance_matrix)
      distance_matrix = distance_matrix.detach().cpu().numpy()

      matched_indices, matched_mask = self.greedy_match(distance_matrix)
      #print('matched_indices: ', matched_indices)

      regression_loss = []
      for i in range(matched_indices.shape[0]):
        matched_index = matched_indices[i]
        #print('matched_index: ', matched_index)
        distance = distance_matrix[matched_index[0]][matched_index[1]]
        #print('distance: ', distance)
        if distance < self.gt_data_association_threshold:
          regression_loss_per_pair = self.get_regression_loss_per_pair(
            gt_boxes[matched_index[0]],
            tracking_boxes[matched_index[1]])
          regression_loss.append(regression_loss_per_pair)

      regression_loss_count = len(regression_loss)

      regression_loss = torch.stack(regression_loss, dim=0)
      #print('regression_loss: ', regression_loss)
      regression_loss_sum = torch.sum(regression_loss)
      #print('regression_loss: ', regression_loss)

      return regression_loss_sum, regression_loss_count


    def reset_dkf_gradients(self):
      '''
      Reset each class DKF's 
      class DifferentiableKalmanFilter
        in self.trackers,
      in order to run loss.backward() and optimizer.step 
      multiple times during tracking a sequence,
      by using detach and clone tensor values of DifferentiableKalmanFilter
      '''
      for tracker in self.trackers:
        tracker.reset_gradients()

    def process_dets_to_gt_order(self, dets):
      '''
      Input
        dets - list of  numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
      Output
        dets_in_gt_order: numpy [N, 7], [x, y, z, theta, l, w, h]
      '''
      #print('dets: ', dets)

      if len(dets) == 0:
        return np.zeros([0, 7])

      dets = np.stack(dets, axis=0)
      #print('dets.shape: ', dets.shape)

      dets_in_gt_order = np.concatenate(
        [
          dets[:, 3:7],
          dets[:, 2:3],
          dets[:, 1:2],
          dets[:, 0:1]
        ],
        axis=1
      )
      
      #print('dets_in_gt_order: ', dets_in_gt_order)
      #print('dets_in_gt_order.shape: ', dets_in_gt_order.shape)

      return dets_in_gt_order



    def track_multi_sensor_differentiable_kalman_filter(self, dets_all_dict, frame, seq_name, cav_id_list, dets_feature_dict, gt_boxes, gt_ids, transformation_matrix_dict):
      """
      Params:
        dets_all_dict: dict
          dets_all_dict[cav_id] is a dict: cav_id is in cav_id_list
            dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
            info: a array of other info for each det
          frame:    str, frame number, used to query ego pose
          cav_id_list: match tracks with detection from vehicles in the order of cav_id_list
          dets_feature_dict: object detection feature per object
          gt_boxes: ground truth box [N, 7], [x, y, z, theta, l, w, h] 
          gt_ids: ground truth object id: [N], N: number of objects in this frame
      Requires: this method must be called once for each frame even with empty detections.
      Returns the a similar array, where the last column is the object ID.

      Output:
        results: tracking result boxes
        affi: TODO
        loss_dict: loss dictionary {
          'regression' : {
            'sum' :
            'count' : 
          },
          'association' : {
            'sum' :
            'count' : 
          }
        }
        for each type of loss, we store both sum and count,
        in order to calculate average loss of all tracking boxes
        after finish tracking a sequence

      NOTE: The number of objects returned may differ from the number of detections provided.
      """
      # MY_DEBUG
      # when measuring run time during inference, ignore loss calculation
      measure_run_time = False

      # experiment that change the order
      #cav_id_list = ['1', 'ego']
      loss_dict = {}


      dets_dict = {}
      info_dict = {}
      for cav_id in dets_all_dict.keys():
        dets_dict[cav_id] = dets_all_dict[cav_id]['dets'] # dets: N x 7, float numpy array
        info_dict[cav_id] = dets_all_dict[cav_id]['info']

      if self.debug_id: print('\nframe is %s' % frame)

      # logging
      print_str = '\n\n*****************************************\n\nprocessing seq_name/frame %s/%d' % (seq_name, frame)
      print_log(print_str, log=self.log, display=False)
      self.frame_count += 1

      # recall the last frames of outputs for computing ID correspondences during affinity processing
      self.id_past_output = copy.copy(self.id_now_output)
      self.id_past = [trk.id for trk in self.trackers]

      # my process detection to gt order
      # from  [h,w,l,x,y,z,theta] to [x, y, z, theta, l, w, h]
      dets_in_gt_order_dict = {}
      for cav_id in dets_all_dict.keys():
        dets_in_gt_order_dict[cav_id] = self.process_dets_to_gt_order(dets_dict[cav_id])

      # process detection format
      # from [h,w,l,x,y,z,theta] to Box3D()
      for cav_id in dets_all_dict.keys():
        dets_dict[cav_id] = self.process_dets(dets_dict[cav_id])

        

      default_init_P, _, _ = DKF.get_ab3dmot_default_covariance_matrices(self.dtype, self.device, dim_x=10, dim_z=7)

      # get observation covariance matrix from ObservationCovarianceNet
      observation_covariance_dict = {}
      learnable_init_P_dict = {}
      learnable_R_dict = {}
      det_neg_log_likelihood_loss_dict = {}
      det_neg_log_likelihood_loss_sum = []
      det_neg_log_likelihood_loss_count = 0
      for cav_id in dets_all_dict.keys():
        dets_feature = dets_feature_dict[cav_id]
        dets_feature = torch.tensor(dets_feature, dtype=self.dtype, device=self.device)
        transformation_matrix = transformation_matrix_dict[cav_id]
        transformation_matrix = torch.tensor(transformation_matrix, dtype=self.dtype, device=self.device)
        dets_in_gt_order = dets_in_gt_order_dict[cav_id]
        dets_in_gt_order = torch.tensor(dets_in_gt_order, dtype=self.dtype, device=self.device)
        #print('cav_id: ', cav_id)
        #print('transformation_matrix: ', transformation_matrix)

        #start_time = time.time()
        # if 0 detection (val seq 0007 frame 32), return torch.zeros([0, 10])
        if self.use_multiple_nets:
          observation_covariance_dict[cav_id] = self.observation_covariance_net_dict[cav_id](
            dets_feature, frame, transformation_matrix, dets_in_gt_order)
          # print('observation_covariance_dict[cav_id]: ', observation_covariance_dict[cav_id])
        else:
          # if use only one net, use the ego's model which is set in optimizer
          observation_covariance_dict[cav_id] = self.observation_covariance_net_dict['ego'](
            dets_feature, frame,  transformation_matrix, dets_in_gt_order)

        # get learnable_R_dict
        learnable_init_P_dict[cav_id], learnable_R_dict[cav_id] = self.get_learnable_observation_covariance(default_init_P, observation_covariance_dict[cav_id])
        #end_time = time.time()
        #print('Covariance Net runtime: %f' % (end_time - start_time))

        if not measure_run_time:
          # calculate negative loglikelihood loss
          # between pair of det and gt
          if dets_in_gt_order.shape[0] == 0:
            # no detection # val seq 0007 frame 32
            continue
          det_neg_log_likelihood_loss_dict[cav_id], matched_det_count = get_neg_log_likelihood_loss(
            dets_in_gt_order, learnable_R_dict[cav_id], gt_boxes)
          det_neg_log_likelihood_loss_sum.append(det_neg_log_likelihood_loss_dict[cav_id])
          det_neg_log_likelihood_loss_count += matched_det_count

      # total det_neg_log_likelihood_loss for all cav
      #print('det_neg_log_likelihood_loss_sum: ', det_neg_log_likelihood_loss_sum)
      #print('det_neg_log_likelihood_loss_count: ', det_neg_log_likelihood_loss_count)
      if measure_run_time:
        loss_dict['det_neg_log_likelihood'] = {
          'sum' : torch.zeros(1, dtype=self.dtype, device=self.device),
          'count' : 1 
        }
      else:
        det_neg_log_likelihood_loss_sum = torch.cat(det_neg_log_likelihood_loss_sum, dim=0)
        det_neg_log_likelihood_loss_sum = torch.sum(det_neg_log_likelihood_loss_sum)
        loss_dict['det_neg_log_likelihood'] = {
          'sum' : det_neg_log_likelihood_loss_sum,
          'count' :det_neg_log_likelihood_loss_count.detach().cpu().numpy() 
        }



      # KF prediction step
      # tracks propagation based on velocity
      #start_time = time.time()
      if self.force_gt_as_predicted_track and self.prev_prev_gt_boxes is not None:
        trks = self.transform_gt_as_predicted_track(frame)
      else:
        trks = self.prediction()
      #end_time = time.time()
      #print('Kalman Filter Prediction Step runtime: %f', (end_time - start_time))

      # ego motion compensation, adapt to the current frame of camera coordinate
      if (frame > 0) and (self.ego_com) and (self.oxts is not None):
        # we do not have self.oxts for v2v4real
        assert False
        trks = self.ego_motion_compensation(frame, trks)

      # visualization
      if self.vis and (self.vis_dir is not None):
        assert False
        img = os.path.join(self.img_dir, f'{frame:06d}.png')
        save_path = os.path.join(self.vis_dir, f'{frame:06d}.jpg'); mkdir_if_missing(save_path)
        self.visualization(img, dets, trks, self.calib, self.hw, save_path)

      # get data association loss
      # before the actual mathcing and update step, after the predict step
      # so the track contains the state right before and after the predict step
      # prev state is used to get the matched gt id in the previous frame
      # current state is used to get the distance to detection box to get loss
      if measure_run_time:
        loss_dict['association'] = {
          'sum' : torch.zeros(1, dtype=self.dtype, device=self.device),
          'count' : 1
        }
      else:
        association_loss_sum, association_loss_count = get_association_loss(dets_dict, self.trackers, gt_boxes, gt_ids, self.prev_gt_boxes, self.prev_gt_ids)
        loss_dict['association'] = {
          'sum' : association_loss_sum,
          'count' : association_loss_count
        }

      # For multi_sensor_kalman_filter,
      # perform match and update for each detection set from each vehicle in the order of cav_id_list

      #print('2 cav_id_list: ', cav_id_list)
      for cav_id in cav_id_list:
        #print('cav_id: ', cav_id)
        if cav_id not in dets_dict.keys():
          # this cav does not detect any object in this frame
          continue

        # matching
        trk_innovation_matrix = None
        if self.metric == 'm_dis':
          trk_innovation_matrix = [trk.compute_innovation_matrix().detach().cpu().numpy() for trk in self.trackers]

        
        # this data association is calculated in numpy
        matched, unmatched_dets, unmatched_trks, cost, affi = \
          data_association(dets_dict[cav_id], trks, self.metric, self.thres, self.algm, trk_innovation_matrix)

        self.update(matched, unmatched_trks, dets_dict[cav_id], info_dict[cav_id], 
          learnable_R_dict, frame, cav_id)


        # create and initialise new trackers for unmatched detections
        new_id_list = self.birth(dets_dict[cav_id], info_dict[cav_id], unmatched_dets, frame, cav_id, learnable_init_P_dict)

        trks = self.get_trks_for_match()

      # calculate loss during training
      # regression loss: measureing the difference between tracking results and gt boxes
      if measure_run_time:
        loss_dict['regression'] = {
          'sum' : torch.zeros(1, dtype=self.dtype, device=self.device),
          'count' : 1
        }
      else:
        regression_loss_sum, regression_loss_count = self.get_regression_loss(gt_boxes)
        loss_dict['regression'] = {
          'sum' : regression_loss_sum,
          'count' : regression_loss_count
        }


      # save gt info for next frame's loss calculation
      self.prev_prev_gt_boxes = self.prev_gt_boxes
      self.prev_prev_gt_ids = self.prev_gt_ids
      self.prev_gt_boxes = gt_boxes
      self.prev_gt_ids = gt_ids

      # output existing valid tracks
      results, matched_detection_id_dict, track_P = self.output()

      if len(results) > 0: results = [np.concatenate(results)]                # h,w,l,x,y,z,theta, ID, other info, confidence
      else:                    results = [np.empty((0, 15))]
      self.id_now_output = results[0][:, 7].tolist()                                  # only the active tracks that are outputed

        

      return results, affi, loss_dict, matched_detection_id_dict, learnable_R_dict, track_P, det_neg_log_likelihood_loss_dict
