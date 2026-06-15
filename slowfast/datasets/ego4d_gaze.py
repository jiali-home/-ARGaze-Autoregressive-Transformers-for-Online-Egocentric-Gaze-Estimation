#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from collections import OrderedDict

import cv2
import numpy as np
import csv
import torch
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment
from .gaze_io_sample import parse_gtea_gaze

# Fast image loader for better performance
try:
    from .fast_image_loader import retry_load_images_fast
    _HAS_FAST_LOADER = True
except ImportError:
    _HAS_FAST_LOADER = False

logger = logging.get_logger(__name__)


def _parse_filter_set(value):
    if value is None:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


@DATASET_REGISTRY.register()
class Ego4dgaze(torch.utils.data.Dataset):
    """
    Ego4D video loader. Construct the Ego4D video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping.
    """

    def __init__(self, cfg, mode, num_retries=2):
        """
        Construct the EGO4D video loader with a given csv file.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for Ego4dGaze".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self._clip_keys = []  # relative clip path for each dataset index
        self._clip_windows = OrderedDict()
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

        logger.info("Constructing Eg4dgaze {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:  # use RandAug
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == 'train':
            path_to_file = getattr(self.cfg.DATA, 'PATH_TO_DATA_DIR', 'data/train_ego4d_gaze_stride8.csv')
            if not pathmgr.exists(path_to_file):
                path_to_file = 'data/train_ego4d_gaze_stride8.csv'
        elif self.mode == 'val' or self.mode == 'test':
            path_to_file = getattr(self.cfg.DATA, 'PATH_TO_TEST_DIR', 'data/test_ego4d_gaze_stride8.csv')
            if not pathmgr.exists(path_to_file):
                path_to_file = 'data/test_ego4d_gaze_stride8.csv'
        else:
            raise ValueError(f"Don't support mode {self.mode}.")

        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = dict()
        self._spatial_temporal_idx = []

        clip_windows = OrderedDict()
        with pathmgr.open(path_to_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].startswith('#'):
                    continue
                if row[0] == 'clip_path':
                    continue
                clip_rel_path = row[0].strip()
                if len(row) >= 3:
                    try:
                        start_frame = int(row[1])
                        end_frame = int(row[2])
                        window = (start_frame, end_frame)
                    except ValueError:
                        window = (None, None)
                else:
                    window = (None, None)
                if clip_rel_path == '':
                    continue
                if clip_rel_path not in clip_windows:
                    clip_windows[clip_rel_path] = []
                if window[0] is not None and window[1] is not None:
                    clip_windows[clip_rel_path].append(window)

        if not clip_windows:
            logger.warning(
                f"No structured rows found in {path_to_file}; falling back to line-based parsing."
            )
            with pathmgr.open(path_to_file, "r") as f2:
                for line in f2.read().splitlines():
                    clip = line.strip()
                    if clip == '':
                        continue
                    clip_windows.setdefault(clip, [])

        # Ensure each clip has at least one placeholder window
        for clip, windows in clip_windows.items():
            if not windows:
                clip_windows[clip] = [(None, None)]

        if self.mode == "test":
            clip_filter = _parse_filter_set(getattr(self.cfg.TEST, "CLIP_INDEX_FILTER", ""))
            sequence_filter = _parse_filter_set(getattr(self.cfg.TEST, "SEQUENCE_FILTER", ""))
            if clip_filter or sequence_filter:
                filtered = OrderedDict()
                for clip_idx, (clip_rel_path, windows) in enumerate(clip_windows.items()):
                    video_name = clip_rel_path.split("/")[0] if "/" in clip_rel_path else clip_rel_path
                    keep_clip = (not clip_filter) or str(clip_idx) in clip_filter
                    keep_sequence = (not sequence_filter) or any(
                        token in clip_rel_path or token in video_name for token in sequence_filter
                    )
                    if keep_clip and keep_sequence:
                        filtered[clip_rel_path] = windows
                logger.info(
                    "Applied Ego4D test filter: keeping %d/%d clips.",
                    len(filtered),
                    len(clip_windows),
                )
                clip_windows = filtered

        # Optional: subsample a fraction for quick debugging.
        if self.mode == 'train':
            frac = float(getattr(self.cfg.DATA, 'SUBSAMPLE_TRAIN_FRACTION', 1.0))
        elif self.mode == 'val':
            frac = float(getattr(self.cfg.DATA, 'SUBSAMPLE_VAL_FRACTION', 1.0))
        else:  # test
            frac = float(getattr(self.cfg.DATA, 'SUBSAMPLE_TEST_FRACTION', 1.0))

        strategy = getattr(self.cfg.DATA, 'SUBSET_STRATEGY', "random")
        max_samples = getattr(self.cfg.DATA, 'SUBSET_MAX_SAMPLES', -1)

        if (frac < 1.0 or strategy == "stratified" or max_samples > 0) and len(clip_windows) > 0:
            keys = list(clip_windows.keys())
            rng = random.Random(self.cfg.RNG_SEED)

            if strategy == "stratified":
                # Group by video directory name
                groups = {}
                for k in keys:
                    v_id = k.split('/')[0] if '/' in k else "unknown"
                    if v_id not in groups: groups[v_id] = []
                    groups[v_id].append(k)

                sampled_keys = []
                for v_id in sorted(groups.keys()):
                    group = groups[v_id]
                    n = max(1, int(len(group) * frac))
                    sampled_keys.extend(rng.sample(group, n))
                keys = sampled_keys
                logger.info(f"Stratified subsampling Ego4dgaze {self.mode}: keeping {len(keys)} clips.")
            else:
                n_keep = max(1, int(len(keys) * max(0.0, min(frac, 1.0))))
                keys = rng.sample(keys, n_keep)
                logger.info(f"Random subsampling Ego4dgaze {self.mode}: keeping {len(keys)}/{len(clip_windows)} clips (fraction={frac}).")

            if 0 < max_samples < len(keys):
                keys = rng.sample(keys, max_samples)
                logger.info(f"Applying max samples limit: keeping {max_samples}/{len(keys)} clips.")

            clip_windows = OrderedDict((k, clip_windows[k]) for k in keys)

        for clip_idx, (clip_rel_path, windows) in enumerate(clip_windows.items()):
            full_path = os.path.join(self.cfg.DATA.PATH_PREFIX, 'clips.gaze', clip_rel_path)

            
            for idx in range(self._num_clips):
                self._path_to_videos.append(full_path)
                self._clip_keys.append(clip_rel_path)
                self._spatial_temporal_idx.append(idx)
                self._video_meta[clip_idx * self._num_clips + idx] = {}
            # Preserve ordered windows for lookup
            self._clip_windows[clip_rel_path] = windows

            
        assert (len(self._path_to_videos) > 0), "Failed to load Ego4dgaze split {} from {}".format(self._split_idx, path_to_file)

        # Read gaze label
        logger.info('Loading Gaze Labels...')
        for path in tqdm(self._path_to_videos):
            video_name = path.split('/')[-2]
            if video_name in self._labels.keys():
                pass
            else:
                label_name = video_name + '_frame_label.csv'
                # prefix = os.path.dirname(self.cfg.DATA.PATH_PREFIX)
                with open(os.path.join(f'{self.cfg.DATA.PATH_PREFIX}/gaze_frame_label', label_name), 'r') as f:
                    rows = [list(map(float, row)) for i, row in enumerate(csv.reader(f)) if i > 0]
                self._labels[video_name] = np.array(rows)[:, 1:]  # [x, y, type,] in line with egtea format

        logger.info("Constructing Ego4D dataloader (size: {}) from {}".format(len(self._path_to_videos), path_to_file))

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]  # 256
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]  # 320
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE  # 224
            if short_cycle_idx in [0, 1]:
                crop_size = int(round(self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx] * self.cfg.MULTIGRID.DEFAULT_S))
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(round(float(min_scale) * crop_size / self.cfg.MULTIGRID.DEFAULT_S))

        elif self.mode in ["val", "test"]:
            temporal_sample_index = (self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS)  # = 0
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = ((self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS) if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1)  # = 1
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                # Don't understand why different scale is used when NUM_SPATIAL_CROPS>1
                # if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                # else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
            )  # = (256, 256, 256)
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        sampling_rate = utils.get_random_sampling_rate(self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE, self.cfg.DATA.SAMPLING_RATE)
        # = 8

        # Try to decode and sample a clip from a video OR load pre-extracted frames.
        # If the current source fails, repeatedly find a random replacement.
        clip_rel_path = self._clip_keys[index]
        windows = self._clip_windows.get(clip_rel_path, [(None, None)])

        for i_try in range(self._num_retries):
            frames = None
            frames_idx = None

            video_path = self._path_to_videos[index]
            video_name, clip_name = video_path.split('/')[-2:]

            window_start = window_end = None
            if windows and windows[0][0] is not None and windows[0][1] is not None:
                if self.mode == 'train':
                    window_start, window_end = random.choice(windows)
                else:
                    # Deterministic selection for eval; clamp in case NUM_ENSEMBLE_VIEWS > len(windows)
                    win_idx = int(max(0, min(len(windows) - 1, temporal_sample_index if temporal_sample_index >= 0 else 0)))
                    window_start, window_end = windows[win_idx]

            if window_start is not None and window_end is not None:
                clip_fstart = int(window_start) + 1  # convert 0-based to 1-based for consistency with legacy logic
                clip_fend = int(window_end)
            else:
                clip_tstart, clip_tend = clip_name[:-4].split('_')[-2:]
                clip_tstart, clip_tend = int(clip_tstart[1:]), int(clip_tend[1:])
                clip_fstart = clip_tstart * self.cfg.DATA.TARGET_FPS + 1
                clip_fend = clip_tend * self.cfg.DATA.TARGET_FPS

            frames_root = getattr(self.cfg.DATA, 'FRAMES_DIR', None)


            # Load pre-extracted frames from disk
            # Compute temporal window and sample indices
            video_size = max(1, int(clip_fend - clip_fstart + 1))  # number of frames in this clip
            clip_size = max(1.0, float(sampling_rate) * float(self.cfg.DATA.NUM_FRAMES))

            # Get start/end indices for temporal sampling
            start_idx, end_idx = decoder.get_start_end_idx(
                video_size,
                clip_size,
                temporal_sample_index if self.mode != 'train' else -1,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS if self.mode != 'train' else 1,
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
            )

            # Sample frame indices uniformly
            idx = torch.linspace(start_idx, end_idx, self.cfg.DATA.NUM_FRAMES)
            # idx = torch.clamp(idx, 0, video_size - 1).long()
            idx = torch.arange(self.cfg.DATA.NUM_FRAMES, dtype=torch.float) * sampling_rate + start_idx


            # Map to global frame indices
            frames_global_idx = idx.numpy() + int(clip_fstart) - 1

            # Build image paths and load
            ext = getattr(self.cfg.DATA, 'FRAME_EXT', 'jpg')
            missing_offsets = getattr(self.cfg.DATA, "MISSING_FRAME_OFFSETS", [2, 4, 6, 8])
            max_label_idx = self._labels[video_name].shape[0]

            def _resolve_fid(fid):
                path = os.path.join(frames_root, video_name, f"{int(fid):06d}.{ext}")
                if os.path.exists(path):
                    return int(fid)
                for delta in missing_offsets:
                    fid_next = int(fid) + int(delta)
                    if fid_next >= max_label_idx:
                        continue
                    path_next = os.path.join(frames_root, video_name, f"{fid_next:06d}.{ext}")
                    if os.path.exists(path_next):
                        return fid_next
                return int(fid)

            frames_global_idx = np.array([_resolve_fid(fid) for fid in frames_global_idx], dtype=np.int64)

            img_paths = [
                os.path.join(frames_root, video_name, f"{int(fid):06d}.{ext}") for fid in frames_global_idx
            ]

            # check if all img_paths are valid, if not, skip this sample
            if not all(os.path.exists(img_path) for img_path in img_paths):
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if _HAS_FAST_LOADER:
                # Fast multi-threaded loader with turbojpeg support

                imgs = retry_load_images_fast(img_paths, retry=3, backend='auto', num_workers=4)
                # Returns (T, H, W, 3) in RGB, uint8
            else:
                # Fallback to original loader
                imgs = utils.retry_load_images(img_paths, retry=3, backend="pytorch")  # (T, H, W, 3), BGR
                if isinstance(imgs, torch.Tensor):
                    # Convert BGR->RGB
                    imgs = imgs[:, :, :, [2, 1, 0]]
                else:
                    imgs = torch.as_tensor(np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]))

            frames = imgs  # (T, H, W, 3), uint8
            # Produce frames_idx tensor consistent with decode path
            frames_idx = torch.as_tensor(frames_global_idx, dtype=torch.long) - (int(clip_fstart) - 1)

            # Get gaze labels for sampled frames
            if self.mode not in ['test'] and frames_global_idx[-1] >= self._labels[video_name].shape[0]:  # Some frames don't have labels. Try to use another one
                # logger.info('No annotations:', video_name, clip_name)
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames_global_idx = frames_global_idx.astype(np.int64)
            label = self._labels[video_name][frames_global_idx, :]

            # Filter out samples with invalid gaze_type during training
            # Valid: gaze_type in {0, 1, 2} (fixation, saccade, out-of-bounds)
            # Invalid: gaze_type in {3, 4} (untracked, reserved)
            if self.mode in ['train']:
                if label.shape[1] >= 3:
                    invalid_frames = np.isin(label[:, 2], [3, 4])
                    if np.any(invalid_frames):
                        # Skip this sample - contains invalid GT frames
                        index = random.randint(0, len(self._path_to_videos) - 1)
                        continue

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            if self.aug:
                if self.cfg.AUG.NUM_SAMPLE > 1:

                    frame_list = []
                    label_list = []
                    index_list = []
                    for _ in range(self.cfg.AUG.NUM_SAMPLE):
                        new_frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)
                        label = self._labels[index]
                        new_frames = utils.pack_pathway_output(self.cfg, new_frames)
                        frame_list.append(new_frames)
                        label_list.append(label)
                        index_list.append(index)
                    return frame_list, label_list, index_list, {}

                else:
                    frames = self._aug_frame(frames, spatial_sample_index, min_scale, max_scale, crop_size)

            else:
                frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)
                # Perform data augmentation.
                frames, label = utils.spatial_sampling(
                    frames,
                    gaze_loc=label,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )

            frames = utils.pack_pathway_output(self.cfg, frames)

            label_hm = np.zeros(shape=(frames[0].size(1), frames[0].size(2) // 4, frames[0].size(3) // 4))
            for i in range(label_hm.shape[0]):
                # Check gaze_type validity
                # Valid: gaze_type in {0, 1, 2} (fixation, saccade, out-of-bounds)
                # Invalid: gaze_type in {3, 4} (untracked, reserved)
                gaze_type = int(label[i, 2]) if label.shape[1] >= 3 else 0

                if gaze_type in [3, 4]:
                    # Invalid GT: generate uniform distribution (complete uncertainty)
                    label_hm[i, :, :] = 1.0 / (label_hm.shape[1] * label_hm.shape[2])
                else:
                    # Valid GT: generate Gaussian heatmap
                    self._get_gaussian_map(label_hm[i, :, :], center=(label[i, 0] * label_hm.shape[2], label[i, 1] * label_hm.shape[1]),
                                           kernel_size=self.cfg.DATA.GAUSSIAN_KERNEL, sigma=self.cfg.DATA.HEATMAP_SIGMA)
                    d_sum = label_hm[i, :, :].sum()
                    if d_sum == 0:  # gaze may be outside the image
                        label_hm[i, :, :] = label_hm[i, :, :] + 1 / (label_hm.shape[1] * label_hm.shape[2])
                    elif d_sum != 1:  # gaze may be right at the edge of image
                        label_hm[i, :, :] = label_hm[i, :, :] / d_sum

            label_hm = torch.as_tensor(label_hm).float()
            return frames, label, label_hm, index, {'path': self._path_to_videos[index], 'index': np.array(frames_global_idx)}
        else:
            index = random.randint(0, len(self._path_to_videos) - 1)
            return self.__getitem__(index)

    def _aug_frame(
        self,
        frames,
        spatial_sample_index,
        min_scale,
        max_scale,
        crop_size,
    ):
        aug_transform = create_random_augment(
            input_size=(frames.size(1), frames.size(2)),
            auto_augment=self.cfg.AUG.AA_TYPE,
            interpolation=self.cfg.AUG.INTERPOLATION,
        )
        # T H W C -> T C H W.
        frames = frames.permute(0, 3, 1, 2)
        list_img = self._frame_to_list_img(frames)
        list_img = aug_transform(list_img)
        frames = self._list_img_to_frames(list_img)
        frames = frames.permute(0, 2, 3, 1)

        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE, self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE)
        relative_scales = (None if (self.mode not in ["train"] or len(scl) == 0) else scl)
        relative_aspect = (None if (self.mode not in ["train"] or len(asp) == 0) else asp)
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
            if self.mode in ["train"]
            else False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.cfg.AUG.RE_PROB,
                mode=self.cfg.AUG.RE_MODE,
                max_count=self.cfg.AUG.RE_COUNT,
                num_splits=self.cfg.AUG.RE_COUNT,
                device="cpu",
            )
            frames = frames.permute(1, 0, 2, 3)
            frames = erase_transform(frames)
            frames = frames.permute(1, 0, 2, 3)

        return frames

    def _frame_to_list_img(self, frames):
        img_list = [transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

    @staticmethod
    def _get_gaussian_map(heatmap, center, kernel_size, sigma):
        h, w = heatmap.shape
        mu_x, mu_y = round(center[0]), round(center[1])
        left = max(mu_x - (kernel_size - 1) // 2, 0)
        right = min(mu_x + (kernel_size - 1) // 2, w-1)
        top = max(mu_y - (kernel_size - 1) // 2, 0)
        bottom = min(mu_y + (kernel_size - 1) // 2, h-1)

        if left >= right or top >= bottom:
            pass
        else:
            kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
            kernel_2d = kernel_1d * kernel_1d.T
            k_left = (kernel_size - 1) // 2 - mu_x + left
            k_right = (kernel_size - 1) // 2 + right - mu_x
            k_top = (kernel_size - 1) // 2 - mu_y + top
            k_bottom = (kernel_size - 1) // 2 + bottom - mu_y

            heatmap[top:bottom+1, left:right+1] = kernel_2d[k_top:k_bottom+1, k_left:k_right+1]

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
