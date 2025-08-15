"""
A differentiable, pytorch version of face landmarker alignment to a canonical view (mp_alignment_original.py).
The function align_landmarks() contains code snippets inspired by/in part borrowing from
Rasmus Jones's (Github user Rassibassi) face alignment implementation, found at 
https://github.com/Rassibassi/mediapipeDemos/blob/main/head_posture.py, which itself contains code snippets inspired 
by/in part borrowing from the Google cpp implementation  from github.com/google/mediapipe (copyright below)

Copyright 2020 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from canonical_landmarks import canonical_metric_landmarks, procrustes_landmark_basis
import torch
# import open3d as o3d
from torch import nn


class PCF:
    def __init__(
        self,
        near=1,
        far=10000,
        frame_height=1920,
        frame_width=1080,
        fy=1074.520446598223,
    ):

        self.near = near
        self.far = far
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.fy = fy

        fov_y = 2 * torch.arctan(frame_height / (2 * fy))
        # kDegreesToRadians = np.pi / 180.0 # never used
        height_at_near = 2 * near * torch.tan(0.5 * fov_y)
        width_at_near = frame_width * height_at_near / frame_height

        self.fov_y = fov_y
        self.left = -0.5 * width_at_near
        self.right = 0.5 * width_at_near
        self.bottom = -0.5 * height_at_near
        self.top = 0.5 * height_at_near

class MPAligner(nn.Module):
    def __init__(self):
        super(MPAligner, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.canonical_metric_landmarks = canonical_metric_landmarks.to(self.device)

        self.landmark_weights = torch.zeros((canonical_metric_landmarks.shape[1],), dtype=torch.float32)
        for idx, weight in procrustes_landmark_basis:
            self.landmark_weights[idx] = weight
        self.landmark_weights = self.landmark_weights.to(self.device)


    def get_metric_landmarks(self, screen_landmarks, pcf):
        screen_landmarks = self.project_xy(screen_landmarks, pcf)
        depth_offset = torch.mean(screen_landmarks[2, :])

        intermediate_landmarks = screen_landmarks.clone()
        intermediate_landmarks = self.change_handedness(intermediate_landmarks)
        first_iteration_scale = self.estimate_scale(intermediate_landmarks)

        intermediate_landmarks = screen_landmarks.clone()
        intermediate_landmarks = self.move_and_rescale_z(
            pcf, depth_offset, first_iteration_scale, intermediate_landmarks
        )
        intermediate_landmarks = self.unproject_xy(pcf, intermediate_landmarks)
        intermediate_landmarks = self.change_handedness(intermediate_landmarks)
        second_iteration_scale = self.estimate_scale(intermediate_landmarks)

        metric_landmarks = screen_landmarks.clone()
        total_scale = first_iteration_scale * second_iteration_scale
        metric_landmarks = self.move_and_rescale_z(
            pcf, depth_offset, total_scale, metric_landmarks
        )
        metric_landmarks = self.unproject_xy(pcf, metric_landmarks)
        metric_landmarks = self.change_handedness(metric_landmarks)

        pose_transform_mat = self.solve_weighted_orthogonal_problem(
            self.canonical_metric_landmarks, metric_landmarks, self.landmark_weights
        )
    

        inv_pose_transform_mat = torch.linalg.inv(pose_transform_mat)
        inv_pose_rotation = inv_pose_transform_mat[:3, :3]
        inv_pose_translation = inv_pose_transform_mat[:3, 3]

        #pose_rotation = pose_transform_mat[:3, :3]
        pose_translation = pose_transform_mat[:3, 3]

        metric_landmarks = (
            inv_pose_rotation @ metric_landmarks + inv_pose_translation[:, None]
        )

        return metric_landmarks, pose_transform_mat


    def project_xy(self, landmarks, pcf):
        x_scale = pcf.right - pcf.left
        y_scale = pcf.top - pcf.bottom
        x_translation = pcf.left
        y_translation = pcf.bottom

        landmarks[1, :] = 1.0 - landmarks[1, :]

        landmarks = landmarks * torch.tensor([[x_scale, y_scale, x_scale]]).T.to(self.device)
        landmarks = landmarks + torch.tensor([[x_translation, y_translation, 0]]).T.to(self.device)
        return landmarks


    def change_handedness(self, landmarks):
        landmarks[2, :] *= -1.0

        return landmarks


    def move_and_rescale_z(self, pcf, depth_offset, scale, landmarks):
        landmarks[2, :] = (landmarks[2, :] - depth_offset + pcf.near) / scale

        return landmarks


    def unproject_xy(self, pcf, landmarks):
        landmark_depths = landmarks[2, :]
        unprojected_landmarks = landmarks.clone()
        unprojected_landmarks[0, :] = landmarks[0, :] * landmark_depths / pcf.near
        unprojected_landmarks[1, :] = landmarks[1, :] * landmark_depths / pcf.near

        return unprojected_landmarks


    def estimate_scale(self, landmarks):
        transform_mat = self.solve_weighted_orthogonal_problem(
            self.canonical_metric_landmarks, landmarks, self.landmark_weights
        )

        return torch.linalg.norm(transform_mat[:, 0])


    def extract_square_root(self, point_weights):
        return torch.sqrt(point_weights).to(torch.float32)


    def solve_weighted_orthogonal_problem(self, source_points, target_points, point_weights):
        sqrt_weights = self.extract_square_root(point_weights)
        transform_mat = self.internal_solve_weighted_orthogonal_problem(
            source_points, target_points, sqrt_weights
        )
        return transform_mat


    def internal_solve_weighted_orthogonal_problem(self, sources, targets, sqrt_weights):

        # tranposed(A_w).
        weighted_sources = sources * sqrt_weights[None, :]
        # tranposed(B_w).
        weighted_targets = targets * sqrt_weights[None, :]
        # w = tranposed(j_w) j_w.
        total_weight = torch.sum(sqrt_weights * sqrt_weights)

        # Let C = (j_w tranposed(j_w)) / (tranposed(j_w) j_w).
        # Note that C = tranposed(C), hence (I - C) = tranposed(I - C).
        #
        # tranposed(A_w) C = tranposed(A_w) j_w tranposed(j_w) / w =
        # (tranposed(A_w) j_w) tranposed(j_w) / w = c_w tranposed(j_w),
        #
        # where c_w = tranposed(A_w) j_w / w is a k x 1 vector calculated here:
        twice_weighted_sources = weighted_sources * sqrt_weights[None, :]
        source_center_of_mass = torch.sum(twice_weighted_sources, axis=1) / total_weight
    
        # tranposed((I - C) A_w) = tranposed(A_w) (I - C) =
        # tranposed(A_w) - tranposed(A_w) C = tranposed(A_w) - c_w tranposed(j_w).
        centered_weighted_sources = weighted_sources - torch.matmul(
            source_center_of_mass[:, None], sqrt_weights[None, :]
        )

        design_matrix = torch.matmul(weighted_targets, centered_weighted_sources.T)


        rotation = self.compute_optimal_rotation(design_matrix)

        scale = self.compute_optimal_scale(
            centered_weighted_sources, weighted_sources, weighted_targets, rotation
        )

        rotation_and_scale = scale * rotation

        pointwise_diffs = weighted_targets - torch.matmul(rotation_and_scale, weighted_sources)


        weighted_pointwise_diffs = pointwise_diffs * sqrt_weights[None, :]

        translation = torch.sum(weighted_pointwise_diffs, axis=1) / total_weight


        transform_mat = self.combine_transform_matrix(rotation_and_scale, translation)


        return transform_mat


    def compute_optimal_rotation(self, design_matrix):
        if torch.linalg.norm(design_matrix) < 1e-9:
            print("Design matrix norm is too small!")

        u, _, vh = torch.linalg.svd(design_matrix, full_matrices=True)

        postrotation = u
        prerotation = vh

        if torch.linalg.det(postrotation) * torch.linalg.det(prerotation) < 0:
            postrotation[:, 2] = -1 * postrotation[:, 2]


        rotation = torch.matmul(postrotation, prerotation)


        return rotation


    def compute_optimal_scale(self,
        centered_weighted_sources, weighted_sources, weighted_targets, rotation
    ):
        rotated_centered_weighted_sources = torch.matmul(rotation, centered_weighted_sources)

        numerator = torch.sum(rotated_centered_weighted_sources * weighted_targets)
        denominator = torch.sum(centered_weighted_sources * weighted_sources)

        if denominator < 1e-9:
            print("Scale expression denominator is too small!")
        if numerator / denominator < 1e-9:
            print("Scale is too small!")

        return numerator / denominator


    def combine_transform_matrix(self, r_and_s, t):
        result = torch.eye(4).to(self.device)
        result[:3, :3] = r_and_s
        result[:3, 3] = t
        return result



    def forward(self, landmarks, init_width, init_height, curr_width, curr_height, z=-50, use_all_landmarks=False, refine_landmarks=True):
        #init_width/height are the dimensions of the input video frame, and curr_width/height are the dimensions of the frame passed to 
        #the facial landmark extractor, which could be a crop (if initial face detection is being used). It is important to use
        # curr_width, curr_height for the intrinsic matrix/PCF used in the transformations. init_width/height should be used
        # to project the aligned 3D landmarks to an image of the same resolution as the input video frame. 
        # NOTE: landmark_coords_2d_aligned should only be used for annotation!  

        points_idx = [33, 263, 61, 291, 199]
        points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
        points_idx = list(set(points_idx))
        points_idx.sort()

        if use_all_landmarks: 
            points_idx = list(range(0,468))
            points_idx[0:2] = points_idx[0:2:-1]

        # pseudo camera internals
        focal_length = curr_width
        center = (curr_width / 2, curr_height / 2)
        camera_matrix = torch.Tensor(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        ).to(torch.float32)

        pcf = PCF(
            near=1,
            far=10000,
            frame_height=curr_height,
            frame_width=curr_width,
            fy=camera_matrix[1, 1],
        )

        landmarks[:, 0] = landmarks[:, 0] / curr_width
        landmarks[:, 1] = landmarks[:, 1] / curr_height
        landmarks = landmarks.T

        if refine_landmarks:
            landmarks = landmarks[:, :468]
        
        landmarks_copy = landmarks.clone()
        metric_landmarks, pose_transform_mat = self.get_metric_landmarks(
            landmarks_copy, pcf
        )
        metric_landmarks = metric_landmarks.T
        
        # Reprojection in numpy, can be used to confirm that my new version below in torcxh is correct
        # import numpy as np
        # import cv2
        # dist_coeff = np.zeros((4, 1))
        # init_focal_length = init_width
        # init_center = (init_width / 2, init_height / 2)
        # init_camera_matrix = np.array(
        #     [[init_focal_length, 0, init_center[0]], [0, init_focal_length, init_center[1]], [0, 0, 1]],
        #     dtype = "double")
        # no_rot = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # no_rotV, _ = cv2.Rodrigues(no_rot)
        # translation = np.float32([[0], [0], [z]])
        # metric_landmarks_np = metric_landmarks.clone().detach().numpy()
        # world_points_2d, _ = cv2.projectPoints(
        #     metric_landmarks_np.T,
        #     no_rotV,
        #     translation,
        #     init_camera_matrix,
        #     dist_coeff,
        # )

        # projection of the 3D metric landmarks under no distortion, rotation, and translation only in z dimension is simply 
        # given by fX/(Z + z) + init_center[0], fY/(Z + z) + init_center[1], where X, Y, Z are the coordinates of the 3D metric landmarks
        # and z is the provided z translation. This can be confirmed by uncommenting the code above and comparing the results with
        # results computed below
        init_focal_length = init_width
        init_center = (init_width / 2, init_height / 2)
        landmark_coords_2d_aligned = metric_landmarks[:, :2] 
        landmark_coords_2d_aligned[:, 0] = ((landmark_coords_2d_aligned[:, 0] * init_focal_length )/(z + metric_landmarks[:,2]))  + init_center[0]
        landmark_coords_2d_aligned[:, 1] = ((landmark_coords_2d_aligned[:, 1] * init_focal_length )/(z + metric_landmarks[:,2]))  + init_center[1]

        return metric_landmarks, landmark_coords_2d_aligned
