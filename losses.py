# Copyright 2022 BBC and University of Surrey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchvision

from utils import NDC2Euclidean

def gradient_x(img):
	gx = img[:,:,:-1,:] - img[:,:,1:,:]
	return gx

def gradient_y(img):
	gy = img[:,:-1,:,:] - img[:,1:,:,:]
	return gy

def get_disparity_smoothness(disp, img):
	"""Disparity smoothness loss

	Smoothness of disparity weighted by the smoothness of the image
	Similar to Monodepth (https://github.com/mrharicot/monodepth)
	"""
	disp_gradients_x = gradient_x(disp)
	disp_gradients_y = gradient_y(disp)

	image_gradients_x = gradient_x(img)
	image_gradients_y = gradient_y(img)

	weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 3, keepdim=True))
	weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 3, keepdim=True))

	smoothness_x = torch.mean(torch.abs(disp_gradients_x) * weights_x)
	smoothness_y = torch.mean(torch.abs(disp_gradients_y) * weights_y)
	return smoothness_x + smoothness_y

def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :-1] - image[:, :, 1:])) + \
        torch.mean(torch.abs(image[:, :-1, :] - image[:, 1:, :]))
    return loss

def distortion_loss(ray_weights, t_vals):
	"""Calculate distortion loss

	From Mip-NeRF 360 (https://github.com/google-research/multinerf)
	\mathcal{L}_{dist}(\mathbf{t}, \mathbf{w}) =
		\sum_{i,j} w_{i} w_{j} \left| \frac{t_{i} + t_{i+1}}{2} - \frac{t_{j} + t_{j+1}}{2} \right|
		+ \frac{1}{3}\sum _{i} w_{i}^{2}( t_{i+1} - t_{i}))

	Args:
		ray_weights: [N, N_rays, N_samples] alpha compositing weights assigned to each sample along a ray
		t_vals:      [N, N_samples] sample positions along a ray. Normalised.
	"""

	# Product of every pair of point weights
	N, N_rays, N_samples = ray_weights.shape
	w_expanded = ray_weights[...,None].expand(-1, -1, N_samples, N_samples)
	w_transpose = w_expanded.transpose(-2, -1)
	w_pairs = w_expanded * w_transpose # [N, N_rays, N_samples, N_samples]

	# Distances between all pairs of interval midpoints
	t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
	pairs_interval_midpoints = torch.abs(t_mids[...,None] - t_mids)

	# Weighted distances
	# \sum_{i,j} w_{i} w_{j} \left| \frac{t_{i} + t_{i+1}}{2} - \frac{t_{j} + t_{j+1}}{2} \right|
	weighted_dist = 0.5 * torch.sum(w_pairs[..., :-1, :-1] * pairs_interval_midpoints, axis=[-1,-2])

	# Weighted size of each individual interval
	w_square = ray_weights * ray_weights
	t_dists = t_vals[..., 1:] - t_vals[..., :-1]
	individual_interval_size = (1/3) * torch.sum(w_square[..., :-1] * t_dists, axis=-1)

	loss = torch.sum(weighted_dist + individual_interval_size)

	return loss

def mse_masked(pred, gt, mask):
	"""Returns the Mean Squared Error of a predicted image in only the masked region

	Inputs:
	- pred: [N,N_rays, 3]
	- gt: [N,N_rays, 3]
	- mask: [N,N_rays, 1]
	"""
	d = [1 for t in range(pred.dim() - 1)] # Num of dimensions to keep as they are
	mask_rep = mask.repeat(*d, pred.size(-1)) # Repeat along the last dimension
	num_pix = torch.sum(mask_rep) + 1e-8
	mse = torch.sum(((pred - gt) ** 2) * mask_rep) / num_pix
	return mse

def mae_masked(pred, gt, mask):
	"""Returns the Mean Absolute Error of a predicted image in only the masked region

	Inputs:
	- pred: [N,N_rays, 3]
	- gt: [N,N_rays, 3]
	- mask: [N,N_rays, 1]
	"""

	d = [1 for t in range(pred.dim() - 1)] # Num of dimensions to keep as they are
	mask_rep = mask.repeat(*d, pred.size(-1)) # Repeat along the last dimension
	num_pix = torch.sum(mask_rep) + 1e-8
	mae = torch.sum(torch.abs(pred - gt) * mask_rep) / num_pix
	return mae

def compute_depth_loss(pred_depth, gt_depth):
	"""Compute scene flow depth loss

	Encourage the expected termination deth computed along each ray
	to be close to the depth predicted from a pre-trained single view network.
	As single-view depth predictions are defined up to an unknown scale and shift,
	we use a robust scale-shift invatian loss where the depth goes through a whitening
	operation that normalises the depth to have zero mean and unit scale
	"""

	# Normalise predicted depth
	t_pred = torch.median(pred_depth)
	s_pred = torch.mean(torch.abs(pred_depth - t_pred))

	# Normalise ground truth depth
	t_gt = torch.median(gt_depth)
	s_gt = torch.mean(torch.abs(gt_depth - t_gt))

	pred_depth_n = (pred_depth - t_pred)/s_pred
	gt_depth_n = (gt_depth - t_gt)/s_gt

	# This is different from the paper NSFF where they say the loss is L1
	return torch.mean(torch.pow(pred_depth_n - gt_depth_n, 2))

def compute_sf_smooth_loss(pts_1_ndc, pts_2_ndc, H, W, f):
	"""Compute scene flow spatial smoothness loss

	Scene flow spatial smoothness minimizes the weighted l1 difference
	between scenes flows sampled at neighboring 3D position along each ray
	"""
	n = pts_1_ndc.shape[-2] # number of samples per ray

	# Discard farthest ray samples?
	pts_1_ndc_close = pts_1_ndc[..., :int(n * 0.95), :]
	pts_2_ndc_close = pts_2_ndc[..., :int(n * 0.95), :]

	pts_3d_1_world = NDC2Euclidean(pts_1_ndc_close, H, W, f)
	pts_3d_2_world = NDC2Euclidean(pts_2_ndc_close, H, W, f)

	# scene flow
	scene_flow_world = pts_3d_1_world - pts_3d_2_world

	return torch.mean(torch.abs(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :]))

# Least kinetic motion prior
def compute_sf_lke_loss(pts_ref_ndc, pts_post_ndc, pts_prev_ndc, H, W, f):
	"""Compute scene flow temporal smoothness loss

	Scene flow temporal smoothness, inspired by Vo et al. [9], encourages 3D point
	trajectories to be piece-wise linear with least kinetic energy prior. This is
	equivalent to minimizing sum of forward scene flow and backward scene flow from
	each sampled 3D point along the ray

	Inputs:
	- pts_ref_ndc: [N, N_rays, N_samples, 3] ndc rays at time t
	- pts_post_ndc: [N, N_rays, N_samples, 3] ndc rays at time t+1
	calculated using estimated flow fields
	- pts_prev_ndc: [N, N_rays, N_samples, 3] ndc rays at time t-1
	calculated using estimated flow fields
	- H: image height
	- W: image width
	- f: camera focal point
	"""
	n = pts_ref_ndc.shape[-2]

	pts_ref_ndc_close = pts_ref_ndc[..., :int(n * 0.9), :]
	pts_post_ndc_close = pts_post_ndc[..., :int(n * 0.9), :]
	pts_prev_ndc_close = pts_prev_ndc[..., :int(n * 0.9), :]

	pts_3d_ref_world = NDC2Euclidean(pts_ref_ndc_close, H, W, f)
	pts_3d_post_world = NDC2Euclidean(pts_post_ndc_close, H, W, f)
	pts_3d_prev_world = NDC2Euclidean(pts_prev_ndc_close, H, W, f)

	# scene flow
	scene_flow_w_ref2post = pts_3d_post_world - pts_3d_ref_world
	scene_flow_w_prev2ref = pts_3d_ref_world - pts_3d_prev_world

	if False:
		# Visualisation
		cat_img = torch.concat([pts_ref_ndc_close, pts_post_ndc_close, pts_prev_ndc_close,
								pts_3d_ref_world, pts_3d_post_world, pts_3d_prev_world,
								scene_flow_w_ref2post, scene_flow_w_prev2ref])
		torchvision.utils.save_image(cat_img.permute(0, 3, 1, 2), "vis_sceneflow_lke/cat_img3.png")
		exit()

	return 0.5 * torch.mean((scene_flow_w_ref2post - scene_flow_w_prev2ref) ** 2)
