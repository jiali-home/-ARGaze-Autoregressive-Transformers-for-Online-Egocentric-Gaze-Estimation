#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import bisect

import ipdb
import cv2
import numpy as np
import torch
import torch.utils.data
import pickle
import time
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

logger = logging.get_logger(__name__)


def _parse_filter_set(value):
    if value is None:
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


@DATASET_REGISTRY.register()
class Egteagaze(torch.utils.data.Dataset):
    """
    EGTEA Gaze video loader. Construct the EGTEA video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the EGTEA video loader with a given csv file.
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set.
                For the test mode, the data loader will take data from test set.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for EgteaGaze".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._frame_id_cache = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)

        logger.info("Constructing Egteagaze {}...".format(mode))
        self._construct_loader()
        self.aug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:  # use RandAug
            self.aug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _get_frame_ids(self, frames_root, video_name, ext):
        cache_key = (frames_root, video_name, ext)
        if cache_key in self._frame_id_cache:
            return self._frame_id_cache[cache_key]

        frame_dir = os.path.join(frames_root, video_name)
        frame_ids = []
        try:
            for fname in os.listdir(frame_dir):
                stem, suffix = os.path.splitext(fname)
                if suffix.lower() == f".{ext.lower()}" and stem.isdigit():
                    frame_ids.append(int(stem))
        except Exception:
            frame_ids = []
        frame_ids.sort()
        self._frame_id_cache[cache_key] = frame_ids
        return frame_ids

    def _resolve_frame_path(self, frames_root, video_name, frame_id, ext):
        path = os.path.join(frames_root, video_name, f"{frame_id:06d}.{ext}")
        if pathmgr.exists(path):
            return path

        # EGTEA clip names use one-based F indices, while gaze labels are
        # indexed from zero. Prefer the label index, then fall back to the
        # frame filename convention used by the extracted frame directory.
        alt_path = os.path.join(frames_root, video_name, f"{frame_id + 1:06d}.{ext}")
        if pathmgr.exists(alt_path):
            return alt_path

        frame_ids = self._get_frame_ids(frames_root, video_name, ext)
        if not frame_ids:
            return path

        pos = bisect.bisect_left(frame_ids, frame_id)
        candidates = []
        if pos < len(frame_ids):
            candidates.append(frame_ids[pos])
        if pos > 0:
            candidates.append(frame_ids[pos - 1])
        nearest = min(candidates, key=lambda fid: (abs(fid - frame_id), fid))
        return os.path.join(frames_root, video_name, f"{nearest:06d}.{ext}")

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == 'train':
            path_to_file = getattr(self.cfg.DATA, 'PATH_TO_DATA_DIR', 'data/train_gaze_official.csv')
            if not pathmgr.exists(path_to_file):
                path_to_file = 'data/train_gaze_official.csv'
        elif self.mode in ['val', 'test']:
            path_to_file = getattr(self.cfg.DATA, 'PATH_TO_TEST_DIR', 'data/test_gaze_official.csv')
            if not pathmgr.exists(path_to_file):
                path_to_file = 'data/test_gaze_official.csv'
        else:
            raise ValueError(f"Dont't support mode {self.mode}.")

        assert pathmgr.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = dict()
        self._spatial_temporal_idx = []
        with pathmgr.open(path_to_file, "r") as f:
            paths = [item for item in f.read().splitlines()
                     if self.mode != 'test' or 'OP03-R01-PastaSalad-879780-892210-F021084-F021444.mp4' not in item]  # In test set, label doesn't cover this clip

            if self.mode == "test":
                clip_filter = _parse_filter_set(getattr(self.cfg.TEST, "CLIP_INDEX_FILTER", ""))
                sequence_filter = _parse_filter_set(getattr(self.cfg.TEST, "SEQUENCE_FILTER", ""))
                if clip_filter or sequence_filter:
                    filtered_paths = []
                    for clip_idx, path in enumerate(paths):
                        video_name = path.split("/")[0] if "/" in path else os.path.basename(path)
                        keep_clip = (not clip_filter) or str(clip_idx) in clip_filter
                        keep_sequence = (not sequence_filter) or any(
                            token in path or token in video_name for token in sequence_filter
                        )
                        if keep_clip and keep_sequence:
                            filtered_paths.append(path)
                    logger.info(
                        "Applied EGTEA test filter: keeping %d/%d clips.",
                        len(filtered_paths),
                        len(paths),
                    )
                    paths = filtered_paths

            # Optional: subsample a fraction for quick debugging.
            if self.mode == 'train':
                frac = float(getattr(self.cfg.DATA, 'SUBSAMPLE_TRAIN_FRACTION', 1.0))
            elif self.mode == 'val':
                frac = float(getattr(self.cfg.DATA, 'SUBSAMPLE_VAL_FRACTION', 1.0))
            else:  # test
                frac = float(getattr(self.cfg.DATA, 'SUBSAMPLE_TEST_FRACTION', 1.0))

            strategy = getattr(self.cfg.DATA, 'SUBSET_STRATEGY', "random")
            max_samples = getattr(self.cfg.DATA, 'SUBSET_MAX_SAMPLES', -1)

            if (frac < 1.0 or strategy == "stratified" or max_samples > 0) and len(paths) > 0:
                rng = random.Random(self.cfg.RNG_SEED)
                if strategy == "stratified":
                    # Group by (Subject, Activity) for diversity
                    groups = {}
                    for p in paths:
                        basename = os.path.basename(p)
                        parts = basename.split('-')
                        key = "-".join(parts[:3]) if len(parts) >= 3 else "unknown"
                        if key not in groups: groups[key] = []
                        groups[key].append(p)

                    sampled_paths = []
                    for key in sorted(groups.keys()):
                        group = groups[key]
                        n = max(1, int(len(group) * frac))
                        sampled_paths.extend(rng.sample(group, n))
                    paths = sampled_paths
                    logger.info(f"Stratified subsampling Egteagaze {self.mode}: keeping {len(paths)}/{len(paths)} clips.")
                else:
                    # Random sampling
                    n_keep = max(1, int(len(paths) * max(0.0, min(frac, 1.0))))
                    paths = rng.sample(paths, n_keep) if n_keep < len(paths) else paths
                    logger.info(f"Random subsampling Egteagaze {self.mode}: keeping {n_keep}/{len(paths)} clips (fraction={frac}).")

                # Apply absolute limit if set
                if 0 < max_samples < len(paths):
                    paths = rng.sample(paths, max_samples)
                    logger.info(f"Applying max samples limit: keeping {max_samples}/{len(paths)} clips.")

            for clip_idx, path in enumerate(paths):
                for idx in range(self._num_clips):
                    self._path_to_videos.append(os.path.join(self.cfg.DATA.PATH_PREFIX, 'cropped_clips', path))
                    self._spatial_temporal_idx.append(idx)  # used in test
                    self._video_meta[clip_idx * self._num_clips + idx] = {}  # only used in torchvision backend
        assert (len(self._path_to_videos) > 0), "Failed to load Egteagaze split {} from {}".format(self._split_idx, path_to_file)

        if self.mode == 'train':  # self._spatial_temporal_idx is not used in training, only shuffle paths
            random.shuffle(self._path_to_videos)

        # Read gaze label with optional caching
        logger.info('Loading Gaze Labels...')
        # Prepare cache directory
        cache_dir_cfg = getattr(self.cfg.DATA, 'GAZE_CACHE_DIR', '')
        if cache_dir_cfg and cache_dir_cfg.strip() != '':
            cache_dir = cache_dir_cfg
        else:
            cache_dir = os.path.join(self.cfg.DATA.PATH_PREFIX, 'gaze_cache')
        use_cache = bool(getattr(self.cfg.DATA, 'GAZE_CACHE_ENABLE', True))
        if use_cache:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception:
                use_cache = False

        for path in tqdm(self._path_to_videos):
            video_name = path.split('/')[-2]
            if video_name in self._labels:
                continue
            label_name = video_name + '.txt'
            label_path = os.path.join(f'{self.cfg.DATA.PATH_PREFIX}/gaze_data', label_name)

            # Try cache
            cached = False
            if use_cache:
                cache_path = os.path.join(cache_dir, f'{video_name}.pt')
                try:
                    if os.path.exists(cache_path):
                        src_mtime = os.path.getmtime(label_path) if os.path.exists(label_path) else 0
                        cache_mtime = os.path.getmtime(cache_path)
                        if cache_mtime >= src_mtime:
                            # Load torch cache
                            obj = torch.load(cache_path, map_location='cpu')
                            if isinstance(obj, dict) and 'labels' in obj:
                                arr = obj['labels']
                            else:
                                arr = obj
                            if isinstance(arr, torch.Tensor):
                                arr = arr.numpy()
                            self._labels[video_name] = arr
                            cached = True
                except Exception:
                    cached = False

            if not cached:
                print(f"Parsing gaze label for {video_name} from {label_path}")
                arr = parse_gtea_gaze(label_path)
                self._labels[video_name] = arr
                if use_cache:
                    try:
                        torch.save({'labels': torch.from_numpy(arr)}, cache_path)
                    except Exception:
                        pass

        logger.info("Constructing egteagaze dataloader (size: {}) from {}".format(len(self._path_to_videos), path_to_file))

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
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )  # = 1
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
        for i_try in range(self._num_retries):
            frames = None
            frames_idx = None

            video_path = self._path_to_videos[index]
            video_name, clip_name = video_path.split('/')[-2:]
            clip_fstart, clip_fend = clip_name[:-4].split('-')[-2:]  # get start and end frame indices
            clip_fstart, clip_fend = int(clip_fstart[1:]), int(clip_fend[1:])  # remove 'F'

            use_frame_dir = False
            frames_root = getattr(self.cfg.DATA, 'FRAMES_DIR', None)
            if frames_root and pathmgr.exists(os.path.join(frames_root, video_name)):
                use_frame_dir = True

            if use_frame_dir:
                # Compute deterministic/random temporal window within the clip, then sample T indices equally spaced.
                try:
                    from .decoder import get_start_end_idx
                except Exception:
                    get_start_end_idx = None

                video_size = max(1, clip_fend - clip_fstart + 1)  # number of frames in this clip
                # Approximate clip_size in frame units; assume fps ~ target_fps
                clip_size = max(1.0, float(sampling_rate) * float(self.cfg.DATA.NUM_FRAMES))
                if get_start_end_idx is not None:
                    start_idx, end_idx = decoder.get_start_end_idx(
                        video_size,
                        clip_size,
                        temporal_sample_index if self.mode != 'train' else -1,
                        self.cfg.TEST.NUM_ENSEMBLE_VIEWS if self.mode != 'train' else 1,
                        use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                    )
                else:
                    # Fallback: center window
                    delta = max(video_size - clip_size, 0)
                    start_idx = delta / 2.0
                    end_idx = start_idx + clip_size - 1

                idx = torch.linspace(start_idx, end_idx, self.cfg.DATA.NUM_FRAMES)
                idx = torch.clamp(idx, 0, video_size - 1).long()
                # Map to global frame indices
                frames_global_idx = idx.numpy() + clip_fstart - 1

                # Build image paths and load
                ext = getattr(self.cfg.DATA, 'FRAME_EXT', 'jpg')
                img_paths = [
                    self._resolve_frame_path(frames_root, video_name, int(fid), ext)
                    for fid in frames_global_idx
                ]

                try:
                    imgs = utils.retry_load_images(img_paths, retry=3, backend="pytorch")  # (T, H, W, 3), BGR
                    if isinstance(imgs, torch.Tensor):
                        # Convert BGR->RGB
                        imgs = imgs[:, :, :, [2, 1, 0]]
                    else:
                        imgs = torch.as_tensor(np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]))
                    frames = imgs  # (T, H, W, 3), uint8
                    # Produce a frames_idx tensor consistent with decode path
                    frames_idx = torch.as_tensor(frames_global_idx, dtype=torch.long) - (clip_fstart - 1)
                except Exception as e:
                    # logger.warning(f"Failed to load frames for {video_name} from dir; fallback to video decode. Err: {e}")
                    use_frame_dir = False

            if not use_frame_dir:
                # Fallback: decode from video file
                video_container = None
                try:
                    video_container = container.get_video_container(
                        video_path,
                        self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                        self.cfg.DATA.DECODING_BACKEND,
                    )
                except Exception as e:
                    logger.info("Failed to load video from {} with error {}".format(video_path, e))

                # Select a random video if the current video was not able to access.
                if video_container is None:
                    logger.warning("Failed to meta load video idx {} from {}; trial {}".format(index, video_path, i_try))
                    if self.mode not in ["test"] and i_try > self._num_retries // 2:
                        # let's try another one
                        index = random.randint(0, len(self._path_to_videos) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                frames, frames_idx = decoder.decode(
                    container=video_container,
                    sampling_rate=sampling_rate,
                    num_frames=self.cfg.DATA.NUM_FRAMES,
                    clip_idx=temporal_sample_index,
                    num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self._video_meta[index],
                    target_fps=self.cfg.DATA.TARGET_FPS,
                    backend=self.cfg.DATA.DECODING_BACKEND,
                    max_spatial_scale=min_scale,  # only used in torchvision backend
                    use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                    get_frame_idx=True
                )

                # Map to global indices for labels
                if frames_idx is not None:
                    frames_global_idx = frames_idx.numpy() + clip_fstart - 1

            # Get gaze label for sampled frames
            if frames is None:
                logger.warning("Failed to fetch frames for idx {} from {}; trial {}".format(index, video_path, i_try))
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Some frames don't have labels. Try to use another one during training.
            if self.mode not in ['test']:
                if frames_global_idx[-1] > self._labels[video_name].shape[0]:
                    index = random.randint(0, len(self._path_to_videos) - 1)
                    continue
            label = self._labels[video_name][frames_global_idx, :]
            label[:, 0][np.where(label[:, 2] == 0)] = 0.5  # In untracked frame, set gaze at the center initially. It will be covered by a uniform distribution.
            label[:, 1][np.where(label[:, 2] == 0)] = 0.5

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
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

            # label_hm = np.zeros(shape=(frames[0].size(1), frames[0].size(2), frames[0].size(3)))
            label_hm = np.zeros(shape=(frames[0].size(1), frames[0].size(2) // 4, frames[0].size(3) // 4))
            for i in range(label_hm.shape[0]):
                if label[i, 2] == 0:  # if gaze is untracked, use uniform distribution
                    label_hm[i, :, :] = label_hm[i, :, :] + 1 / (label_hm.shape[1] * label_hm.shape[2])
                else:
                    self._get_gaussian_map(
                        label_hm[i, :, :],
                        center=(label[i, 0] * label_hm.shape[2], label[i, 1] * label_hm.shape[1]),
                        kernel_size=self.cfg.DATA.GAUSSIAN_KERNEL,
                        sigma=self.cfg.DATA.HEATMAP_SIGMA,
                    )
                d_sum = label_hm[i, :, :].sum()
                if d_sum == 0:  # gaze may be outside the image
                    label_hm[i, :, :] = label_hm[i, :, :] + 1 / (label_hm.shape[1] * label_hm.shape[2])
                elif d_sum != 1:  # gaze may be right at the edge of image
                    label_hm[i, :, :] = label_hm[i, :, :] / d_sum

            label_hm = torch.as_tensor(label_hm).float()
            return frames, label, label_hm, index, {'path': self._path_to_videos[index], 'index': np.array(frames_global_idx)}
        else:
            raise RuntimeError("Failed to fetch video after {} retries.".format(self._num_retries))

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
            motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT if self.mode in ["train"] else False,
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


# --------------------------- Visualization Utils --------------------------- #
def _select_frame_indices(T: int, k: int):
    if k <= 0:
        return []
    if k >= T:
        return list(range(T))
    xs = np.linspace(0, T - 1, num=k)
    return sorted(list({int(round(v)) for v in xs}))


def _to_rgb_uint8(img_float_chw: torch.Tensor):
    """
    Convert CHW float [0,1] tensor to HxWx3 RGB uint8 numpy array.
    """
    img = img_float_chw.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return img


def _overlay_heatmap_rgb(img_rgb_uint8: np.ndarray, heatmap_2d_float: np.ndarray, alpha=0.45):
    h, w, _ = img_rgb_uint8.shape
    hm = heatmap_2d_float.astype(np.float32)
    # Normalize heatmap for visualization
    hm = hm - float(hm.min())
    denom = float(hm.max())
    hm = hm / (denom + 1e-6)
    hm_resized = cv2.resize(hm, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_vis = (hm_resized * 255.0).clip(0, 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)  # BGR
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb_uint8, 1.0, hm_color, alpha, 0)
    return blended


def visualize_sample(cfg, mode="train", index=0, num_frames_to_show=4, save_dir=None, fname_prefix="egtea_sample", log_to_wandb=False, wb_run=None):
    """
    Visualize a dataset sample by overlaying GT heatmaps on raw frames.

    Args:
        cfg: config with DATA.MEAN/STD and OUTPUT_DIR.
        mode: "train" or "val" or "test".
        index: dataset index to visualize.
        num_frames_to_show: how many frames to sample and show.
        save_dir: directory to save images; default to <OUTPUT_DIR>/egtea_preview.
        fname_prefix: file name prefix when saving.
        log_to_wandb: if True, log images to wandb (requires wb_run or active wandb.init).
        wb_run: optional wandb run object.

    Returns:
        list of numpy arrays (RGB uint8) for panels saved/logged.
    """
    # Lazy import wandb if requested
    wandb = None
    if log_to_wandb:
        try:
            import wandb as _wandb  # type: ignore
            wandb = _wandb
        except Exception:
            wandb = None

    dataset = Egteagaze(cfg, mode)
    frames, labels, labels_hm, _, meta = dataset[index]

    # frames: list of tensors or tensor. Use the single pathway.
    if isinstance(frames, (list,)):
        x = frames[0]
    else:
        x = frames
    # x: [C, T, H, W]
    C, T, H, W = x.shape
    # Denormalize to [0,1] for CHW format
    mean = torch.tensor(cfg.DATA.MEAN, device=x.device, dtype=x.dtype).view(C, 1, 1, 1)
    std = torch.tensor(cfg.DATA.STD, device=x.device, dtype=x.dtype).view(C, 1, 1, 1)
    x_denorm = (x * std + mean).clamp(0.0, 1.0)

    # labels_hm: [T, Hh, Wh]
    assert labels_hm.dim() == 3, f"labels_hm should be [T,H,W], got {tuple(labels_hm.shape)}"

    frame_indices = _select_frame_indices(T, int(max(1, num_frames_to_show)))
    panels = []
    for t in frame_indices:
        frame_rgb = _to_rgb_uint8(x_denorm[:, t])
        gt_hm = labels_hm[t].detach().cpu().numpy()
        gt_overlay = _overlay_heatmap_rgb(frame_rgb, gt_hm, alpha=0.45)

        spacer = np.ones((frame_rgb.shape[0], 8, 3), dtype=np.uint8) * 255
        panel = np.concatenate([frame_rgb, spacer, gt_overlay], axis=1)
        panels.append(panel)

    if len(panels) == 0:
        return []

    # Save outputs
    if save_dir is None or save_dir == "":
        save_dir = os.path.join(cfg.OUTPUT_DIR, "egtea_preview")
    os.makedirs(save_dir, exist_ok=True)
    out_paths = []
    for i, img in enumerate(panels):
        out_path = os.path.join(save_dir, f"{fname_prefix}_{mode}_{index}_t{i}.jpg")
        # cv2.imwrite expects BGR
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out_paths.append(out_path)

    # Optionally log to wandb
    if log_to_wandb and (wandb is not None):
        images = [wandb.Image(panels[i], caption=f"{mode} idx={index} t={frame_indices[i]}") for i in range(len(panels))]
        (wb_run or wandb).log({f"{mode}/dataset_preview": images})

    return panels
