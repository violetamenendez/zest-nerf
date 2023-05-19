from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
from PIL import Image
from torchvision import transforms as T
from kornia import create_meshgrid

from .data_utils import center_poses


class NSFFDataset(Dataset):
    def __init__(self, root_dir, config_dir, split='train', crossval='NSFF',
                 downSample=1.0, max_len=-1,
                 scene=None, closest_views=False,
                 use_mvs=False, use_mvs_dy=False,
                 num_keyframes=10, frame_jump=1,
                 render_spiral=False, target_idx=10,
                 img_h=288, img_w=544):
        """
        Neural Scene Flow Fields dataset https://github.com/zhengqili/Neural-Scene-Flow-Fields
        """
        self.root_dir = Path(root_dir)
        self.config_dir = Path(config_dir)
        self.split = split
        self.crossval = crossval
        self.use_mvs = use_mvs
        self.use_mvs_dy = use_mvs_dy
        self.num_keyframes = num_keyframes
        self.frame_jump = frame_jump # Number of frames to skip to get the temporal neighbours
        self.downSample = downSample
        self.img_wh = (int(img_w*downSample),int(img_h*downSample))
        assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
            'image width must be divisible by 32, you may need to modify the imgScale'
        self.max_len = max_len
        self.define_transforms()
        self.blender2opencv = np.array([[1, 0, 0, 0],
                                        [0,-1, 0, 0],
                                        [0, 0,-1, 0],
                                        [0, 0, 0, 1]])

        self.build_metas(scene)

        self.build_proj_mats()

        self.white_back = False

        self.scale_factor = 1.0 / 200 # scale factor for DTU depth maps


    def build_metas(self, scene):
        if scene == None:
            scene_list = self.config_dir / f'lists/{self.crossval}_{self.split}.txt'

            with open(scene_list) as f:
                self.scenes = [line.rstrip() for line in f.readlines()]
        else:
            self.scenes = [scene]

        self.image_paths, self.disp_paths, self.mask_paths = {}, {}, {}
        self.flow_fwd_paths, self.flow_bwd_paths = {}, {}
        self.metas = []
        self.key_frames = {}
        for scene in self.scenes:
            scene_path = self.root_dir / scene
            self.image_paths[scene] = sorted(scene_path.glob('**/images/*'))
            self.disp_paths[scene] = sorted(scene_path.glob('**/disp/*'))
            self.mask_paths[scene] = sorted(scene_path.glob('**/motion_masks/*'))
            self.flow_fwd_paths[scene] = sorted(scene_path.glob('**/flow_i1/*_fwd.npz'))
            self.flow_bwd_paths[scene] = sorted(scene_path.glob('**/flow_i1/*_bwd.npz'))

            num_frames = len(self.image_paths[scene])
            for frame_t in range(num_frames):
                self.metas += [(scene, frame_t, num_frames)]

            self.key_frames[scene] = []
            interval = num_frames // (self.num_keyframes - 1)
            for frame_id in range(0, num_frames, interval):
                self.key_frames[scene].append(frame_id)

    def build_proj_mats(self):
        """
        For all the dataset, cam00.mp4 is the center reference camera which
        we held out for testing.

        The poses_bounds.npy store all the camera poses correspond to their
        sequential sorted order in the folder. If cam03.mp4 does not exist in
        the final folder, it does not contain the pose for that. The length
        of total cameras in the poses_bounds.npy file should be equal to the
        total number of valid video streams.
        """
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = {}, {}, {}, {}
        self.wander_path = {} # Rendering
        self.poses, self.bounds = {}, {}
        self.N_images = 0
        for scene in self.scenes:

            scene_path = self.root_dir / scene
            poses_bounds = np.load(scene_path / 'dense' / 'poses_bounds.npy') # For each scene

            if self.split in ['train', 'val']:
                assert len(poses_bounds) == len(self.image_paths[scene]), \
                    f'Mismatch between number of images {len(self.image_paths[scene])} and ' \
                        f'number of poses {len(poses_bounds)} in {scene}! Please rerun COLMAP!'

            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
            bounds = poses_bounds[:, -2:]  # (N_images, 2)

            # Step 1: rescale focal length according to training resolution
            H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

            focal = [focal* self.img_wh[0] / W, focal* self.img_wh[1] / H]

            # Step 2: correct poses
            poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            poses, _ = center_poses(poses, self.blender2opencv)

            # Step 3: correct scale so that the nearest depth is at a little more than 1.0
            scale_factor = np.percentile(bounds[:, 0], 5) * 0.9  # 0.75 is the default parameter, 0.9 from NSFF
            bounds /= scale_factor
            poses[..., 3] /= scale_factor
            self.bounds[scene] = bounds
            self.poses[scene] = poses

            proj_mats_scene = []
            intrinsics_scene = []
            world2cams_scene = []
            cam2worlds_scene = []
            wanderpath_scene = []

            w, h = self.img_wh
            num_frames = len(poses)
            for idx in range(num_frames):
                # camera-to-world, world-to-camera
                c2w = torch.eye(4).float()
                c2w[:3] = torch.FloatTensor(poses[idx])
                w2c = torch.inverse(c2w)
                cam2worlds_scene.append(c2w)
                world2cams_scene.append(w2c)

                # Intrisics are the same for all views
                intrinsic = torch.tensor([[focal[0], 0, w / 2], [0, focal[1], h / 2], [0, 0, 1]]).float()
                intrinsics_scene.append(intrinsic.clone())
                intrinsic[:2] = intrinsic[:2] / 4   # 4 times downscale in the feature space

                # Projection matrices
                proj_mat_l = torch.eye(4)
                proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
                proj_mats_scene.append(proj_mat_l)

                # Novel view - wander path
                target_c2w = self.wanderpath_poses(c2w, H, W, focal)
                wanderpath_scene.append(target_c2w)

            self.proj_mats[scene] = torch.stack(proj_mats_scene).float()
            self.intrinsics[scene] = torch.stack(intrinsics_scene).float()
            self.world2cams[scene] = torch.stack(world2cams_scene).float()
            self.cam2worlds[scene] = torch.stack(cam2worlds_scene).float()
            self.wander_path[scene] = torch.stack(wanderpath_scene).float()

    def wanderpath_poses(c2w,  H, W, focal):
        """
        Camera poses to render around a target view

        Inputs:
        - c2w: camera-to-world matrix for the target view
        """
        # TODO - transform this to torch tensors instead of np
        num_frames = 60 # generate 60 frames around one original camera
        max_disp = 48.0 # 64 , 48

        max_trans = max_disp / focal # TODO double check this, it was hwf[2][0] # Maximum camera translation to satisfy max_disp parameter
        output_poses = []

        for i in range(num_frames):
            x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
            y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0 #* 3.0 / 4.0
            z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0

            i_pose = np.concatenate([
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
            ],axis=0)#[np.newaxis, :, :]

            i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

            ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

            render_pose = np.dot(ref_pose, i_pose)
            print('render_pose ', render_pose.shape)
            print(render_pose)
            exit()
            output_poses.append(render_pose)

        return output_poses

    # def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    #     render_poses = []
    #     rads = np.array(list(rads) + [1.])
    #     hwf = c2w[:,4:5]

    #     for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
    #         c = np.dot(c2w[:3,:4],
    #                     np.array([np.cos(theta),
    #                              -np.sin(theta),
    #                              -np.sin(theta*zrate),
    #                               1.]) * rads)

    #         z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
    #         render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    #     return render_poses

    def define_transforms(self):
        self.transform = T.ToTensor()
        self.src_transform = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                                        ])

    def read_optical_flow(self, flow_path, img_wh):

        data = np.load(flow_path)#, (w, h))
        flow, mask = data['flow'], data['mask']
        mask = np.float32(mask)
        flow = cv2.resize(flow, img_wh)
        mask = cv2.resize(mask, img_wh,
                          interpolation=cv2.INTER_NEAREST)

        return flow, mask

    def read_disp(self, filename, img_wh):

        disp = np.load(filename)
        disp = cv2.resize(disp, img_wh,
                          interpolation=cv2.INTER_NEAREST)

        return disp

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        scene, target_frame, num_frames = self.metas[idx]
        print(f"Selected target {scene}, {target_frame}, {num_frames}")

        # all frames for MVS volume + target_frame
        view_ids = []
        if self.use_mvs:
            view_ids += self.key_frames[scene]
        view_ids += [target_frame]
        # First neighbours
        first_nb_ids = [max(target_frame - (1 * self.frame_jump), 0), # +-1 frames
                        min(target_frame + (1 * self.frame_jump), int(num_frames) - 1)]


        imgs, disps, proj_mats = [], [], []
        intrinsics, c2ws, w2cs = [],[],[]
        flow_fwds, flow_bwds = [], []
        mask_fwds, mask_bwds = [], []
        near_fars = []
        near_far_source = torch.Tensor([self.bounds[scene][view_ids].min()*0.8, self.bounds[scene][view_ids].max()*1.2])

        # Poses for first neighbours +-1
        fnb_w2cs = []
        for i, vid in enumerate(first_nb_ids):
            fnb_w2cs.append(self.world2cams[scene][vid])

        if self.use_mvs_dy:
            # Second neighbours
            neighbourgs = [max(target_frame - (2 * self.frame_jump), 0),
                           max(target_frame - (1 * self.frame_jump), 0),
                           min(target_frame + (1 * self.frame_jump), int(num_frames) - 1),
                           min(target_frame + (2 * self.frame_jump), int(num_frames) - 1)]

            # Matrices for  neighbours +-2 frames
            nb_imgs, nb_proj_mats = [], []
            nb_intr, nb_c2ws, nb_w2cs = [],[],[]
            for i, vid in enumerate(neighbourgs):
                nb_intr.append(self.intrinsics[scene][vid])
                nb_w2cs.append(self.world2cams[scene][vid])
                nb_c2ws.append(self.cam2worlds[scene][vid])

                # Is this necessary for neighbours?
                proj_mat_ls = self.proj_mats[scene][vid]
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                nb_proj_mats += [proj_mat_ls @ ref_proj_inv]

                # Get image
                img = Image.open(self.image_paths[scene][vid]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                nb_imgs.append(self.src_transform(img))

        # Matrices for reference view
        for i, vid in enumerate(view_ids):
                intrinsics.append(self.intrinsics[scene][vid])
                w2cs.append(self.world2cams[scene][vid])
                c2ws.append(self.cam2worlds[scene][vid])
                near_fars.append(near_far_source)

                proj_mat_ls = self.proj_mats[scene][vid]
                if i == 0:  # reference view
                    ref_proj_inv = np.linalg.inv(proj_mat_ls)
                    proj_mats += [np.eye(4)]
                else:
                    proj_mats += [proj_mat_ls @ ref_proj_inv]

                # Get image
                img = Image.open(self.image_paths[scene][vid]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                imgs.append(self.src_transform(img))

        # Optical flow for target frame
        if target_frame == 0:
            # First frame has only forwards flow field
            flow_fwd_path = self.flow_fwd_paths[scene][target_frame]
            flow_fwd, fwd_mask = self.read_optical_flow(flow_fwd_path, self.img_wh)
            flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
        elif target_frame == num_frames - 1:
            # Last frame has only backwards flow field
            flow_bwd_path = self.flow_bwd_paths[scene][target_frame - 1]
            flow_bwd, bwd_mask = self.read_optical_flow(flow_bwd_path, self.img_wh)
            flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
        else:
            flow_fwd_path = self.flow_fwd_paths[scene][target_frame]
            flow_bwd_path = self.flow_bwd_paths[scene][target_frame - 1]
            flow_fwd, fwd_mask = self.read_optical_flow(flow_fwd_path, self.img_wh)
            flow_bwd, bwd_mask = self.read_optical_flow(flow_bwd_path, self.img_wh)
        # Correct flow values according to coordinates
        uv_grid = create_meshgrid(self.img_wh[1], self.img_wh[0], normalized_coordinates=False)[0].numpy()
        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        flow_fwds.append(flow_fwd)
        flow_bwds.append(flow_bwd)
        mask_fwds.append(fwd_mask)
        mask_bwds.append(bwd_mask)

        # Disparity for target frame
        disp_path = self.disp_paths[scene][target_frame]
        disp = self.read_disp(disp_path, self.img_wh)
        disps.append(disp)

        # Motion mask for target frame
        mask_path = self.mask_paths[scene][target_frame]
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(self.img_wh, Image.NEAREST)
        mask = self.transform(mask)
        mask = (mask > 1e-3).float().squeeze()
        coords = torch.where(mask > 0.1)
        motion_coords = torch.stack(coords, -1).float()

        sample = {}
        sample['images'] = torch.stack(imgs).float()  # (V, 3, H, W)
        sample['depths'] = torch.from_numpy(np.stack(disps)).float() # (1, H, W)
        sample['flow_fwds'] = torch.from_numpy(np.stack(flow_fwds)).float().permute(0, 3, 1, 2) # (1, 2, H, W)
        sample['flow_bwds'] = torch.from_numpy(np.stack(flow_bwds)).float().permute(0, 3, 1, 2) # (1, 2, H, W)
        sample['mask_fwds'] = torch.from_numpy(np.stack(mask_fwds)).float() # (1, H, W)
        sample['mask_bwds'] = torch.from_numpy(np.stack(mask_bwds)).float() # (1, H, W)
        sample['motion_coords'] = motion_coords # (M, 2) e.g., torch.Size([132079, 2])
        sample['w2cs'] = torch.stack(w2cs).float()  # (V, 4, 4)
        sample['c2ws'] = torch.stack(c2ws).float()  # (V, 4, 4)
        sample['near_fars'] = torch.stack(near_fars).float() # (V, 2)
        sample['proj_mats'] = torch.from_numpy(np.stack(proj_mats)[:,:3]).float() # (V, 3, 4)
        sample['intrinsics'] = torch.stack(intrinsics).float()  # (V, 3, 3)
        sample['time'] = target_frame
        sample['total_frames'] = num_frames

        # First neighbours
        sample['fnb_w2cs'] = torch.stack(fnb_w2cs).float()  # (V, 4, 4)

        if self.use_mvs_dy:
            # Neighbours
            sample['nb_imgs'] = torch.stack(nb_imgs).float()  # (V, 3, H, W)
            sample['nb_w2cs'] = torch.stack(nb_w2cs).float()  # (V, 4, 4)
            sample['nb_c2ws'] = torch.stack(nb_c2ws).float()  # (V, 4, 4)
            sample['nb_intr'] = torch.stack(nb_intr).float()  # (V, 3, 3)
            sample['nb_proj_mats'] = torch.from_numpy(np.stack(nb_proj_mats)[:,:3]).float() # (V, 3, 4)

        return sample