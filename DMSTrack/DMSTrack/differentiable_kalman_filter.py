import numpy as np
import torch
from filterpy.kalman import KalmanFilter

class Filter(object):
    def __init__(self, bbox3D, info, ID):
        self.initial_pos = bbox3D
        self.time_since_update = 0
        self.id = ID
        self.hits = 1  # number of total hits including the first detection
        self.info = info  # other information associated

class DKF(Filter):
    def __init__(self, bbox3D, info, ID, dtype, device, use_static_default_R, frame, 
                cav_id, det_id_in_cav, learnable_init_P):
        '''
        Init a new track with the detection information
        '''
        super().__init__(bbox3D, info, ID)

        self.dtype = dtype
        self.device = device
        self.use_static_default_R = use_static_default_R

        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz 
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        F = np.array([
            [1,0,0,0,0,0,0,1,0,0],  # state transition matrix, dim_x * dim_x
            [0,1,0,0,0,0,0,0,1,0],
            [0,0,1,0,0,0,0,0,0,1],
            [0,0,0,1,0,0,0,0,0,0],  
            [0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1]
        ])     
        F = torch.tensor(F, dtype=self.dtype, device=self.device)

        # measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        H = np.array([
            [1,0,0,0,0,0,0,0,0,0],      
            [0,1,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0]
        ])
        H = torch.tensor(H, dtype=self.dtype, device=self.device)

        # Get default covariance matrices
        P, Q, R = self.get_ab3dmot_default_covariance_matrices(self.dtype, self.device, dim_x=10, dim_z=7)
        self.default_P = P.detach().clone()
        self.default_Q = Q.detach().clone()
        self.default_R = R.detach().clone()

        # Use learnable initial covariance if specified
        if not self.use_static_default_R:
            P = learnable_init_P

        # Set initial velocity to 0
        initial_state_mean = np.concatenate([self.initial_pos.reshape((7, 1)), np.array([[0], [0], [0]])], axis=0)
        initial_state_mean = torch.tensor(initial_state_mean, dtype=self.dtype, device=self.device)
        
        self.dkf = DifferentiableKalmanFilter(F, H, initial_state_mean, P, Q, R, use_static_default_R)

        self.last_updated_frame = frame
        self.last_updated_cav_id = cav_id

        self.matched_detection_id_dict = {'ego': -1, '1': -1}
        self.matched_detection_id_dict[cav_id] = det_id_in_cav
    
    def reset_matched_detection_id_dict(self):
        self.matched_detection_id_dict = {'ego': -1, '1': -1}

    def reset_gradients(self):
        self.dkf.x = self.dkf.x.detach().clone()
        self.dkf.P = self.dkf.P.detach().clone()
        self.dkf.Q = self.dkf.Q.detach().clone()
        self.dkf.R = self.dkf.R.detach().clone()

    @staticmethod
    def get_ab3dmot_default_covariance_matrices(dtype, device, dim_x, dim_z):
        P = torch.eye(dim_x, dtype=dtype, device=device)
        Q = torch.eye(dim_x, dtype=dtype, device=device)
        R = torch.eye(dim_z, dtype=dtype, device=device)

        # Process uncertainty - make velocity components more certain
        Q[7:, 7:] *= 0.01

        return P, Q, R

    def compute_innovation_matrix(self):
        """Compute the innovation matrix for association with mahalanobis distance"""
        return torch.matmul(torch.matmul(self.dkf.H, self.dkf.P), self.dkf.H.t()) + self.dkf.R


class DifferentiableKalmanFilter(object):
    '''
    PyTorch simplified version of filterpy.kalman
    '''
    def __init__(self, F, H, x, P, Q, R, use_static_default_R):
        '''
        torch:
          F: process model matrix
          H: observation model matrix
          x: init state mean
          P: init state covariance matrix
          Q: process model noise covariance
          R: observation model noise covariance
        '''
        self.dtype = P.dtype
        self.device = P.device
        self.dim_x = F.shape[0]
        self.dim_z = H.shape[0]

        self.F = F
        self.H = H
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R
        self.init_P = P.clone()
        self.use_static_default_R = use_static_default_R

    def predict(self, ego_position=None):
        self.prev_x = self.x.clone()
        self.prev_P = self.P.clone()

        self.x = torch.matmul(self.F, self.x)
        self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.t()) + self.Q

    def update(self, z, learnable_R, ego_position=None):
        z = z.reshape([self.dim_z, 1])
        z = torch.tensor(z, dtype=self.dtype, device=self.device)
        
        identity = torch.eye(self.dim_x, dtype=self.dtype, device=self.device)

        if not self.use_static_default_R:
            self.R = learnable_R

        self.S = torch.matmul(torch.matmul(self.H, self.P), self.H.t()) + self.R
        self.K = torch.matmul(torch.matmul(self.P, self.H.t()), torch.inverse(self.S))

        # Kalman filter update on mean
        diff = z - torch.matmul(self.H, self.x)
        additional = torch.matmul(self.K, diff)
        self.x = self.x + additional
        
        # Update covariance
        self.P = torch.matmul(identity - torch.matmul(self.K, self.H), self.P)