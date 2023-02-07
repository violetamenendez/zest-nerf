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
        self.img_wh = (int(960*downSample),int(640*downSample))
        assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
            'image width must be divisible by 32, you may need to modify the imgScale'
        self.max_len = max_len
        self.closest_views = closest_views
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
            poses_bounds = np.load(scene_path / 'poses_bounds.npy') # For each scene

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
            near_original = bounds.min()
            scale_factor = near_original * 0.75  # 0.75 is the default parameter
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

    def read_depth(self, filename, img_wh):
        print(filename, img_wh)
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        print("depth_h shape", depth_h.shape)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        print("depth_h shape", depth_h.shape)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        print("depth_h shape", depth_h.shape)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        print("depth_h shape", depth_h.shape)
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        mask = depth > 0

        depth_h = cv2.resize(depth_h, img_wh)
        print("depth_h shape", depth_h.shape)

        return depth, mask, depth_h

    def read_optical_flow(flow_path):

        data = np.load(flow_path)#, (w, h))
        flow, mask = data['flow'], data['mask']
        mask = np.float32(mask)
        return flow, mask

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        scene, target_frame, num_frames = self.metas[idx]
        print(f"Selected target {scene}, {target_frame}, {num_frames}")

        # Returns a list of all camera poses ordered from nearest to farthest
        nearest_pose_ids = get_nearest_pose_ids(self.cam2worlds[scene][target_frame],
                                                self.cam2worlds[scene],
                                                #5,
                                                len(self.cam2worlds[scene]),
                                                tar_id=target_frame,
                                                angular_dist_method='dist')

        if self.closest_views:
            # Get nearest views to the target image
            nearest_pose_ids = nearest_pose_ids[:5]
        else:
            # Get far views with re. target image
            nearest_pose_ids = nearest_pose_ids[-8:]

        # Select views
        if self.split=='train':
            ids = torch.randperm(5)[:3]
            view_ids = [nearest_pose_ids[i] for i in ids] + [target_frame]
        else:
            view_ids = [nearest_pose_ids[i] for i in range(3)] + [target_frame]
        print(f"Selecting cam views {view_ids}")

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        depths_h = []
        flow_fwds, flow_bwds = [], []
        mask_fwds, mask_bwds = [], []
        near_fars = []
        near_far_source = torch.Tensor([self.bounds[scene][view_ids].min()*0.8, self.bounds[scene][view_ids].max()*1.2])

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

                # Get depth map
                if len(self.depth_files) > 0:
                    depth_filename = random.choice(self.depth_files)
                    depth, mask, depth_h = self.read_depth(depth_filename, self.img_wh)
                    depth_h *= self.scale_factor
                    depths_h.append(depth_h)
                else:
                    depths_h.append(np.zeros((self.img_wh[1], self.img_wh[0])))

                # Optical flow
                if self.optical_flow:
                    if target_frame == 0:
                        # First frame has only forwards flow field
                        flow_fwd_path = self.flow_fwd_paths[scene][target_frame]
                        flow_fwd, fwd_mask = self.read_optical_flow(flow_fwd_path)
                        flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
                    elif target_frame == num_frames - 1:
                        # Last frame has only backwards flow field
                        flow_bwd_path = self.flow_bwd_paths[scene][target_frame - 1]
                        flow_bwd, bwd_mask = self.read_optical_flow(flow_bwd_path)
                        flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
                    else:
                        flow_fwd_path = self.flow_fwd_paths[scene][target_frame]
                        flow_bwd_path = self.flow_bwd_paths[scene][target_frame - 1]
                        flow_fwd, fwd_mask = self.read_optical_flow(flow_fwd_path)
                        flow_bwd, bwd_mask = self.read_optical_flow(flow_bwd_path)
                    flow_fwds.append(flow_fwd)
                    flow_bwds.append(flow_bwd)
                    mask_fwds.append(fwd_mask)
                    mask_bwds.append(bwd_mask)

        sample = {}
        sample['images'] = torch.stack(imgs).float()  # (V, 3, H, W)
        sample['depths_h'] = torch.from_numpy(np.stack(depths_h)).float() # (V, H, W)
        sample['flow_fwds'] = torch.from_numpy(np.stack(flow_fwds)).float() # (V, H, W)
        sample['flow_bwds'] = torch.from_numpy(np.stack(flow_bwds)).float() # (V, H, W)
        sample['mask_fwds'] = torch.from_numpy(np.stack(mask_fwds)).float() # (V, H, W)
        sample['mask_bwds'] = torch.from_numpy(np.stack(mask_bwds)).float() # (V, H, W)
        sample['w2cs'] = torch.stack(w2cs).float()  # (V, 4, 4)
        sample['c2ws'] = torch.stack(c2ws).float()  # (V, 4, 4)
        sample['near_fars'] = torch.stack(near_fars).float()
        sample['proj_mats'] = torch.from_numpy(np.stack(proj_mats)[:,:3]).float()
        sample['intrinsics'] = torch.stack(intrinsics).float()  # (V, 3, 3)
        sample['time'] = target_frame
        sample['total_frames'] = num_frames

        return sample