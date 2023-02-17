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

import logging

import torch
import torch.nn.functional as F

from utils import index_point_feature, build_color_volume

def compute_2d_prob(weights_p_mix, raw_prob_ref2p):
    """ Weight the blending weights of a frame according to the
        estimated confidence computed by the reference frame.
    Inputs:
        weights_p_mix: [N, N_rays, N_samples]. Weights assigned to each sampled color.
        raw_prob_ref2p: [N, N_rays, N_samples]. Estimated confidence in scene flow?

    Returns:
    """
    prob_map_p = torch.sum(weights_p_mix.detach() * (1.0 - raw_prob_ref2p), -1)
    return prob_map_p

def gen_dir_feature(w2c_ref, rays_dir):
    """
    Inputs:
        w2c_ref: [N,4,4]
        rays_dir: [N, N_rays, 3]

    Returns:
        dirs: [N, N_rays, 3]
    """
    logging.info("GEN DIR FEATURE")
    logging.info("inputs "+str(w2c_ref.shape)+","+str(rays_dir.shape))

    dirs = rays_dir @ w2c_ref[:,:3,:3].transpose(1,2) # [N, N_rays, 3]

    logging.info("outputs "+str(dirs.shape))
    return dirs

def gen_pts_feats(imgs, volume_feature, rays_pts, pose_ref, rays_ndc, feat_dim, img_feat=None, img_downscale=1.0, use_color_volume=False, net_type='v0'):
    logging.info("GEN PTS FEATS")
    logging.info("inputs "+str(imgs.shape)+","+str(volume_feature.shape)+"," \
        +str(rays_pts.shape) +","+str(rays_ndc.shape)+","+str(feat_dim)+"," \
        +str(img_feat.shape if img_feat is not None else "None")+"," \
        +str(img_downscale)+","+str(use_color_volume)+","+str(net_type))

    N, N_rays, N_samples = rays_pts.shape[:3]

    if img_feat is not None:
        feat_dim += img_feat.shape[2]*img_feat.shape[3]

    if not use_color_volume:
        input_feat = torch.empty((N, N_rays, N_samples, feat_dim), device=imgs.device, dtype=torch.float)
        ray_feats = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)
        input_feat[..., :8] = ray_feats
        input_feat[..., 8:] = build_color_volume(rays_pts, pose_ref, imgs, img_feat, with_mask=True, downscale=img_downscale)
    else:
        input_feat = index_point_feature(volume_feature, rays_ndc) if torch.is_tensor(volume_feature) else volume_feature(rays_ndc)

    logging.info("outputs "+str(input_feat.shape))
    return input_feat

def depth2dist(z_vals, cos_angle):
    """Compute 'distance' between each integration point along a ray.
    z_vals: [N_ray N_sample]
    """
    device = z_vals.device

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * cos_angle.unsqueeze(-1)
    return dists

def raw2alpha(sigma):
    """
    Function for computing density from model prediction.
    This value is strictly between [0, 1].
    """
    logging.info("RAW2ALPHA")
    logging.info("inputs "+str(sigma.shape))

    alpha = 1. - torch.exp(-sigma)

    # Transmission
    # Compute weight for RGB of each sample along each ray.
    # A cumprod() is used to express the idea of the ray
    # not having reflected up to this point yet.
    # [N_rays, N_samples]
    T = torch.cumprod(torch.cat([torch.ones(*alpha.shape[:2], 1).to(alpha.device),
                                 1. - alpha + 1e-10], -1), -1)[..., :-1]
    # Alpha composite weights
    weights = alpha * T  # [N, N_rays, N_samples]

    logging.info("outputs "+str(alpha.shape)+","+str(weights.shape))
    return alpha, weights

def raw2outputs(raw, z_vals, dists, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [N, N_rays, N_samples, 4]. Prediction from model.
        z_vals: [N, N_rays, N_samples]. Integration time. (depth_candidates)
        dists:  [N, N_rays, N_samples]. Distances. (depth2dist) (t_{i+1}-t_i)
    Returns:
        rgb_map:   [N, N_rays, 3]. Estimated RGB color of a ray.
        disp_map:  [N, N_rays]. Disparity map. Inverse of depth map.
        acc_map:   [N, N_rays]. Sum of weights along each ray.
        weights:   [N, N_rays, N_samples]. Weights assigned to each sampled color.
        depth_map: [N, N_rays]. Estimated distance to object.
    """
    logging.info("RAW2OUTPUTS")
    logging.info("inputs "+str(raw.shape)+","+str(z_vals.shape)+"," \
        +str(dists.shape)+","+str(white_bkgd))

    device = z_vals.device

    rgb = raw[..., :3] # [N, N_rays, N_samples, 3] rgb for each sample point along the ray

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha, weights = raw2alpha(raw[..., 3])  # [N, N_rays, N_samples]

    # Computed weighted color of each sample along each ray.
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N, N_rays, 3]

    # Estimated depth map is expected distance.
    depth_map = torch.sum(weights * z_vals, -1) # [N, N_rays]

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map, device=device), depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, -1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    logging.info("outputs "+str(rgb_map.shape)+","+str(disp_map.shape)+"," \
        +str(acc_map.shape)+","+str(weights.shape)+","+str(depth_map.shape)+","+str(alpha.shape))
    return rgb_map, disp_map, acc_map, weights, depth_map, alpha

def raw2outputs_blending(raw_dy, raw_rigid, raw_blend_w, z_vals, dists, raw_noise_std=0):
    """Transforms dynamic and rigid model's predictions to semantically meaningful values,
       blending the results according to the estimated blending weights.
    Args:
        raw_dy: [N, N_rays, N_samples, 4]. Prediction from dynamic model.
        raw_rigid: [N, N_rays, N_samples, 4]. Prediction from static model.
        raw_blend_w: [N, N_rays, N_samples]. Predicted blending weights.
        z_vals: [N, N_rays, N_samples]. Integration time. (depth_candidates)
        dists:  [N, N_rays, N_samples]. Distances. (depth2dist) (t_{i+1}-t_i)
    Returns:
        rgb_map:   [N, N_rays, 3]. Estimated RGB color of a ray.
        disp_map:  [N, N_rays]. Disparity map. Inverse of depth map.
        acc_map:   [N, N_rays]. Sum of weights along each ray.
        weights:   [N, N_rays, N_samples]. Weights assigned to each sampled color.
        depth_map: [N, N_rays]. Estimated distance to object.
    """
    device = z_vals.device

    rgb_dy = raw_dy[..., :3] # [N_rays, N_samples, 3]
    rgb_rigid = raw_rigid[..., :3] # [N_rays, N_samples, 3]

    # noise = 0.
    # if raw_noise_std > 0.:
    #     noise = torch.randn(raw_dy[..., 3].shape) * raw_noise_std
    # act_fn = F.relu
    # opacity_dy = act_fn(raw_dy[..., 3] + noise)#.detach() #* raw_blend_w
    # opacity_rigid = act_fn(raw_rigid[..., 3] + noise)#.detach() #* (1. - raw_blend_w)
    # NOTE - our opacities already passed an activation function in NeRF. Maybe need to add noise?

    # alpha with blending weights
    alpha_dy = (1. - torch.exp(-raw_dy[..., 3] * dists) ) * raw_blend_w
    alpha_rig = (1. - torch.exp(-raw_rigid[..., 3] * dists)) * (1. - raw_blend_w)

    Ts = torch.cumprod(torch.cat([torch.ones((*alpha_dy.shape[:2], 1), device=device),
                                  (1. - alpha_dy) * (1. - alpha_rig)  + 1e-10], -1), -1)[..., :-1]

    weights_dy = Ts * alpha_dy
    weights_rig = Ts * alpha_rig

    # union map
    rgb_map = torch.sum(weights_dy[..., None] * rgb_dy + \
                        weights_rig[..., None] * rgb_rigid, -2)

    weights_mix = weights_dy + weights_rig
    depth_map = torch.sum(weights_mix * z_vals, -1)

    # compute dynamic depth only
    alpha_fg = 1. - torch.exp(-raw_dy[..., 3] * dists)
    weights_fg = alpha_fg * torch.cumprod(torch.cat([torch.ones((*alpha_fg.shape[:2], 1), device=device),
                                                                 1.-alpha_fg + 1e-10], -1), -1)[..., :-1]
    depth_map_fg = torch.sum(weights_fg * z_vals, -1)
    rgb_map_fg = torch.sum(weights_fg[..., None] * rgb_dy, -2)

    return rgb_map, depth_map, rgb_map_fg, depth_map_fg, weights_fg, weights_dy

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    logging.info("BATCHIFY")
    logging.info("inputs "+str(chunk))

    if chunk is None:
        return fn

    def ret(inputs, alpha_only=False):
        if alpha_only:
            return torch.cat([fn.forward_alpha(inputs[:,i:i + chunk]) for i in range(0, inputs.shape[1], chunk)], 1)
        else:
            return torch.cat([fn(inputs[:,i:i + chunk]) for i in range(0, inputs.shape[1], chunk)], 1)

    return ret

def prepare_pts(args, data_mvs, rays_pts, rays_ndc, rays_dir, cos_angle,
              volume_feature=None, imgs=None, img_feat=None,
              embedding_pts=None, embedding_xyzt=None, embedding_dir=None,
              time_codes=None):
    """Prepares points to be fed to the network.
    Input:
    Output:
        - pts: tensor containing (embedded) rays, time codes, input features, and viewing direction
    """

    # using viewing direction
    if data_mvs is not None:
        w2ref = data_mvs['w2cs'][:,0,:,:]
        angle = gen_dir_feature(w2ref, rays_dir/cos_angle.unsqueeze(-1))  # view dir feature
    else:
        angle = rays_dir/cos_angle.unsqueeze(-1) # [N, N_rays, 1]

    pts = rays_ndc

    # Use point embedder
    if embedding_pts:
        pts = embedding_pts(rays_ndc)
    # Use point embedder for dynamic NeRF (NSFF)
    if embedding_xyzt:
        pts = embedding_xyzt(rays_ndc)

    # Use time codes for Neural 3D video
    if time_codes is not None:
        N, N_rays, N_samples, _ = rays_ndc.shape
        time_codes = time_codes.expand(N, N_rays, N_samples, -1)
        time_codes = F.sigmoid(time_codes)
        pts = torch.cat((pts, time_codes), dim=-1)

    # Use features
    if volume_feature:
        # sample volume feature at ray points
        input_feat = gen_pts_feats(imgs, volume_feature, rays_pts, data_mvs, rays_ndc, args.feat_dim, \
                                img_feat, args.img_downscale, args.use_color_volume, args.net_type)
        if input_feat is not None:
            pts = torch.cat((pts, input_feat), dim=-1)

    # viewing direction
    if angle is not None:
        if angle.dim()!=4:
            # Expand angle (view direction) for each sample
            angle = angle.unsqueeze(dim=2).expand(-1,-1,pts.shape[2],-1)

        if embedding_dir is not None:
            angle = embedding_dir(angle)

        pts = torch.cat([pts, angle], -1)

    return pts, angle is None

def rendering(args, data_mvs, rays_pts, rays_ndc, depth_candidates, rays_dir,
              volume_feature=None, imgs=None, img_feat=None, network_fn=None,
              network_fn_dy=None, embedding_pts=None, embedding_xyzt=None, embedding_dir=None,
              time_codes=None, white_bkgd=False, scene_flow=False):
    """
    rays_dir: [N, N_rays, 3] (e.g. [N,1024,3])
    """
    logging.info("RENDERING")
    logging.info("inputs "+str(rays_pts.shape)+","+str(rays_ndc.shape) \
        +","+str(depth_candidates.shape)+","+str(rays_dir.shape) \
        +","+str(volume_feature.shape if volume_feature is not None else "None") \
        +","+str(imgs.shape if imgs is not None else "None") \
        +","+str(img_feat.shape if img_feat is not None else "None") \
        +","+str("network_fn" if network_fn is not None else "None") \
        +","+str("embedding_pts" if embedding_pts is not None else "None") \
        +","+str("embedding_dir" if embedding_dir is not None else "None") \
        +","+str(time_codes.shape if time_codes is not None else "None") \
        +","+str(white_bkgd))

    device = rays_pts.device

    # rays angle
    cos_angle = torch.norm(rays_dir, dim=-1) # [N, N_rays]

    # Distance between ray intervals
    dists = depth2dist(depth_candidates, cos_angle)

    # Prepare input to the network
    pts, alpha_only = prepare_pts(args, data_mvs, rays_pts, rays_ndc, rays_dir, cos_angle,
                                  volume_feature=volume_feature, imgs=imgs, img_feat=img_feat,
                                  embedding_pts=embedding_pts, embedding_dir=embedding_dir,
                                  time_codes=time_codes)

    # Static NeRF
    outputs_flat = batchify(network_fn, args.netchunk)(pts, alpha_only)

    raw_static = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
    # raw = all the colours and densities of every sample point at every ray

    raw_rgba = raw_static[..., :4]

    rgb_map, disp_map, acc_map, weights, depth_map, alpha = raw2outputs(raw_rgba,
                                                                        depth_candidates,
                                                                        dists, white_bkgd,
                                                                        args.net_type)
    if scene_flow:
        # Blending weights
        raw_blend_w = raw_static[..., 4] #.squeeze()

        # Prepare temporal pts for Dynamic NeRF
        # Add time dimension
        img_idx_rep = torch.ones_like(rays_ndc[..., 0], device=device) * img_idx # NOTE - img_idx is the normalised time [-1,1]
        pts_ref = torch.cat([rays_ndc, img_idx_rep], -1)
        pts_ref_t, _ = prepare_pts(args, data_mvs, rays_pts, pts_ref, rays_dir, cos_angle,
                               volume_feature=volume_feature, imgs=imgs, img_feat=img_feat,
                               embedding_xyzt=embedding_xyzt, embedding_dir=embedding_dir,
                               time_codes=time_codes)
        # TODO - embedding

        # Dynamic NeRF - applied to reference time t
        out_ref_t = batchify(network_fn_dy, args.netchunk)(pts_ref_t) # Apply nerf to points by chunks
        raw_ref_t = torch.reshape(out_ref_t, list(pts_ref_t.shape[:-1]) + [out_ref_t.shape[-1]])
        raw_rgba_ref = raw_ref_t[..., :4] # rgb colour and alpha density
        raw_sf_ref2prev = raw_ref_t[..., 4:7] # scene flow from reference frame to previous frame
        raw_sf_ref2post = raw_ref_t[..., 7:10] # scene flow from reference frame to following frame
        raw_prob_ref2prev = raw_ref_t[:, :, 10] # confidence?
        raw_prob_ref2post = raw_ref_t[:, :, 11] # confidence?

        rgb_map_ref, depth_map_ref, \
        rgb_map_ref_dy, depth_map_ref_dy, \
        weights_ref_dy, weights_ref_dd = raw2outputs_blending(raw_rgba_ref, raw_rgba, raw_blend_w,
                                              depth_candidates, dists, 0)

        weights_map_dd = torch.sum(weights_ref_dd, -1).detach()

        ret = {'rgb_map_ref': rgb_map_ref,
               'depth_map_ref' : depth_map_ref,
               'rgb_map_rig': rgb_map,
               'depth_map_rig': depth_map,
               'rgb_map_ref_dy': rgb_map_ref_dy,
               'depth_map_ref_dy': depth_map_ref_dy,
               'weights_map_dd': weights_map_dd}
        # This is the only necessary returns at inference time. So we need a different rendering for val/test

        # When training we also need:
        ret['raw_sf_ref2prev'] = raw_sf_ref2prev
        ret['raw_sf_ref2post'] = raw_sf_ref2post
        ret['raw_pts_ref'] = pts_ref[..., :3]
        ret['weights_ref_dy'] = weights_ref_dy
        ret['raw_blend_w'] = raw_blend_w
        ret['raw_prob_ref2prev'] = raw_prob_ref2prev
        ret['raw_prob_ref2post'] = raw_prob_ref2post

        ### Previous Frame ###

        # Points for previous frame according to the estimated scene flow + time
        img_idx_rep_prev = torch.ones_like(rays_ndc[..., 0], device=device) * (img_idx - 1./num_img * 2.) # time representation index
        pts_prev = torch.cat([(rays_ndc + raw_sf_ref2prev), img_idx_rep_prev] , -1)
        pts_prev_t, _ = prepare_pts(args, data_mvs, rays_pts, pts_prev, rays_dir, cos_angle,
                                    volume_feature=volume_feature, imgs=imgs, img_feat=img_feat,
                                    embedding_xyzt=embedding_xyzt, embedding_dir=embedding_dir,
                                    time_codes=time_codes)

        # Dynamic NeRF - applied to previous time t - 1
        out_prev_t = batchify(network_fn_dy, args.netchunk)(pts_prev_t)
        raw_prev = torch.reshape(out_prev_t, list(pts_prev_t.shape[:-1]) + [out_prev_t.shape[-1]])
        raw_rgba_prev = raw_prev[:, :, :4]
        raw_sf_prev2prevprev = raw_prev[:, :, 4:7]
        raw_sf_prev2ref = raw_prev[:, :, 7:10]
        ret['raw_pts_prev'] = pts_prev[..., :3]

        # render from t - 1 rgb_map, disp_map, acc_map, weights, depth_map, alpha
        rgb_map_prev_dy, _, _, weights_prev_dy, _, _ = raw2outputs(raw_rgba_prev, depth_candidates, dists)
        ret['raw_sf_prev2ref'] = raw_sf_prev2ref
        ret['rgb_map_prev_dy'] = rgb_map_prev_dy

        ######################

        ##### Next Frame #####

        # Points for the posterior frame according to the estimated scene flow + time
        img_idx_rep_post = torch.ones_like(rays_ndc[..., 0], device=device) * (img_idx + 1./num_img * 2.) # time representation index
        pts_post = torch.cat([(rays_ndc + raw_sf_ref2post), img_idx_rep_post] , -1)
        pts_post_t, _ = prepare_pts(args, data_mvs, rays_pts, pts_post, rays_dir, cos_angle,
                                    volume_feature=volume_feature, imgs=imgs, img_feat=img_feat,
                                    embedding_xyzt=embedding_xyzt, embedding_dir=embedding_dir,
                                    time_codes=time_codes)

        # render points at t + 1
        out_post_t = batchify(network_fn_dy, args.netchunk)(pts_post_t)
        raw_post = torch.reshape(out_post_t, list(pts_post_t.shape[:-1]) + [out_post_t.shape[-1]])
        raw_rgba_post = raw_post[:, :, :4]
        raw_sf_post2ref = raw_post[:, :, 4:7]
        raw_sf_post2postpost = raw_post[:, :, 7:10]
        ret['raw_pts_post'] = pts_post[..., :3]

        rgb_map_post_dy, _, _, weights_post_dy, _, _ = raw2outputs(raw_rgba_post, depth_candidates, dists)
        ret['raw_sf_post2ref'] = raw_sf_post2ref
        ret['rgb_map_post_dy'] = rgb_map_post_dy

        ######################

        # Calculate confidence
        prob_map_prev = compute_2d_prob(weights_prev_dy, raw_prob_ref2prev)
        prob_map_post = compute_2d_prob(weights_post_dy, raw_prob_ref2post)
        ret['prob_map_prev'] = prob_map_prev
        ret['prob_map_post'] = prob_map_post

        ## TODO chain two loss ##


    logging.info("outputs "+str(rgb_map.shape)+","+str(input_feat.shape)+","+str(weights.shape)+","+str(depth_map.shape)+","+str(alpha.shape))
    return rgb_map, input_feat, weights, depth_map, alpha, raw_blend_w
