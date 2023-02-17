from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
from PIL import Image
from torchvision import transforms as T

from .data_utils import get_nearest_pose_ids, center_poses
from utils import read_pfm


class NSFFDataset(Dataset):
    def __init__(self, root_dir, config_dir, split='train',
                 downSample=1.0, max_len=-1,
                 scene=None, closest_views=False):
        """
        Neural Scene Flow Fields dataset https://github.com/zhengqili/Neural-Scene-Flow-Fields
        """
        self.root_dir = Path(root_dir)
        self.config_dir = Path(config_dir)
        self.split = split
        self.downSample = downSample
        self.img_wh = (int(360*downSample),int(480*downSample))
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
            scene_list = self.config_dir / f'lists/NSFF_{self.split}.txt'

            with open(scene_list) as f:
                self.scenes = [line.rstrip() for line in f.readlines()]
        else:
            self.scenes = [scene]

        self.image_paths, self.disp_paths, self.mask_paths = {}, {}, {}
        self.flow_fwd_paths, self.flow_bwd_paths = {}, {}
        self.metas = []
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

            w, h = self.img_wh
            for idx in range(len(poses)):
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
            self.proj_mats[scene] = torch.stack(proj_mats_scene).float()
            self.intrinsics[scene] = torch.stack(intrinsics_scene).float()
            self.world2cams[scene] = torch.stack(world2cams_scene).float()
            self.cam2worlds[scene] = torch.stack(cam2worlds_scene).float()

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

        view_ids = [target_frame]
        # First neighbours
        first_nb_ids = [max(target_frame - 1, 0), # +-1 frames
                        min(target_frame + 1, int(num_frames) - 1)]
        # Second neighbours
        second_nb_ids = [max(target_frame - 2, 0), # +-2 frames
                         min(target_frame + 2, int(num_frames) - 1)]


        imgs, disps, proj_mats = [], [], []
        intrinsics, c2ws, w2cs = [],[],[]
        flow_fwds, flow_bwds = [], []
        mask_fwds, mask_bwds = [], []
        near_fars = []
        near_far_source = torch.Tensor([self.bounds[scene][view_ids].min()*0.8, self.bounds[scene][view_ids].max()*1.2])

        # Matrices for first neighbours +-1 frames
        fnb_imgs, fnb_proj_mats = [], []
        fnb_intr, fnb_c2ws, fnb_w2cs = [],[],[]
        for i, vid in enumerate(first_nb_ids):
            fnb_intr.append(self.intrinsics[scene][vid])
            fnb_w2cs.append(self.world2cams[scene][vid])
            fnb_c2ws.append(self.cam2worlds[scene][vid])

            # Is this necessary for neighbours?
            proj_mat_ls = self.proj_mats[scene][vid]
            ref_proj_inv = np.linalg.inv(proj_mat_ls)
            fnb_proj_mats += [proj_mat_ls @ ref_proj_inv]

            # Get image
            img = Image.open(self.image_paths[scene][vid]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            fnb_imgs.append(self.src_transform(img))

        # Matrices for second neighbours +-2 frames
        snb_imgs, snb_proj_mats = [], []
        snb_intr, snb_c2ws, snb_w2cs = [],[],[]
        for i, vid in enumerate(first_nb_ids):
            snb_intr.append(self.intrinsics[scene][vid])
            snb_w2cs.append(self.world2cams[scene][vid])
            snb_c2ws.append(self.cam2worlds[scene][vid])

            # Is this necessary for neighbours?
            proj_mat_ls = self.proj_mats[scene][vid]
            ref_proj_inv = np.linalg.inv(proj_mat_ls)
            snb_proj_mats += [proj_mat_ls @ ref_proj_inv]

            # Get image
            img = Image.open(self.image_paths[scene][vid]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            snb_imgs.append(self.src_transform(img))

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

                # Optical flow
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
                flow_fwds.append(flow_fwd)
                flow_bwds.append(flow_bwd)
                mask_fwds.append(fwd_mask)
                mask_bwds.append(bwd_mask)

                disp_path = self.disp_paths[scene][target_frame]
                disp = self.read_disp(disp_path, self.img_wh)
                disps.append(disp)

        sample = {}
        sample['images'] = torch.stack(imgs).float()  # (V, 3, H, W)
        sample['depths'] = torch.from_numpy(np.stack(disps)).float() # (V, H, W)
        sample['flow_fwds'] = torch.from_numpy(np.stack(flow_fwds)).float().permute(0, 3, 1, 2) # (V, 2, H, W)
        sample['flow_bwds'] = torch.from_numpy(np.stack(flow_bwds)).float().permute(0, 3, 1, 2) # (V, 2, H, W)
        sample['mask_fwds'] = torch.from_numpy(np.stack(mask_fwds)).float() # (V, H, W)
        sample['mask_bwds'] = torch.from_numpy(np.stack(mask_bwds)).float() # (V, H, W)
        sample['w2cs'] = torch.stack(w2cs).float()  # (V, 4, 4)
        sample['c2ws'] = torch.stack(c2ws).float()  # (V, 4, 4)
        sample['near_fars'] = torch.stack(near_fars).float() # (V, 2)
        sample['proj_mats'] = torch.from_numpy(np.stack(proj_mats)[:,:3]).float() # (V, 3, 4)
        sample['intrinsics'] = torch.stack(intrinsics).float()  # (V, 3, 3)
        sample['time'] = target_frame
        sample['total_frames'] = num_frames

        # First neighbours
        sample['fnb_imgs'] = torch.stack(fnb_imgs).float()  # (V, 3, H, W)
        sample['fnb_w2cs'] = torch.stack(fnb_w2cs).float()  # (V, 4, 4)
        sample['fnb_c2ws'] = torch.stack(fnb_c2ws).float()  # (V, 4, 4)
        sample['fnb_intr'] = torch.stack(fnb_intr).float()  # (V, 3, 3)
        sample['fnb_proj_mats'] = torch.from_numpy(np.stack(fnb_proj_mats)[:,:3]).float() # (V, 3, 4)

        # Second neighbours
        sample['snb_imgs'] = torch.stack(snb_imgs).float()  # (V, 3, H, W)
        sample['snb_w2cs'] = torch.stack(snb_w2cs).float()  # (V, 4, 4)
        sample['snb_c2ws'] = torch.stack(snb_c2ws).float()  # (V, 4, 4)
        sample['snb_intr'] = torch.stack(snb_intr).float()  # (V, 3, 3)
        sample['snb_proj_mats'] = torch.from_numpy(np.stack(snb_proj_mats)[:,:3]).float() # (V, 3, 4)

        return sample