#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime

import ipdb
import numpy as np
import os
from collections import defaultdict, deque
import torch
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.datasets.ava_helper as ava_helper
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)
import torch.nn.functional as F

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(
            cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE
        )
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(
            cfg, mode == "train"
        )
        self.output_dir = cfg.OUTPUT_DIR

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "dt": self.iter_timer.seconds(),
                "dt_data": self.data_timer.seconds(),
                "dt_net": self.net_timer.seconds(),
                "mode": self.mode,
            }
        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            }
            logging.log_json_stats(stats)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_videos,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        if multi_label:
            self.video_preds -= 1e10  # don't understand

        self.video_labels = (
            torch.zeros((num_videos, num_cls))
            if multi_label
            else torch.zeros((num_videos)).long()
        )
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        if self.multi_label:
            self.video_preds -= 1e10
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            if self.video_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.video_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.video_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(
                    self.video_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.video_preds.cpu().numpy(), self.video_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.video_preds, self.video_labels, ks
            )
            topks = [
                (x / self.video_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

            mean_cls_acc, recall_percls = metrics.mean_class_accuracy(preds=self.video_preds, labels=self.video_labels)
            self.stats["mean_clas_acc"] = mean_cls_acc
            self.stats["confusion_matrix"] = metrics.conf_matrix(preds=self.video_preds, labels=self.video_labels).tolist()
        logging.log_json_stats(self.stats)


class TestGazeMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
            self,
            num_videos,
            num_clips,
            num_cls,
            overall_iters,
            dataset,
            store_heatmaps=True,
            edge_threshold=0.0,
            l2_mode="argmax",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options include "sum", and "max".
        """

        self.dataset = dataset
        self.store_heatmaps = store_heatmaps
        self.edge_threshold = edge_threshold
        self.l2_mode = l2_mode
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))

        self.video_labels = torch.zeros((num_videos, num_cls))
        self.clip_count = torch.zeros((num_videos)).long()
        self.topk_accs = []
        self.stats = {}

        self.num_samples = 0.0
        self.recall = list()
        self.precision = list()
        self.f1 = list()
        self.auc = list()
        self.l2 = list()

        self.preds = list()
        self.labels_hm = list()
        self.labels = list()
        self._init_streaming_metrics()

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

        self.num_samples = 0.0
        self.recall = list()
        self.precision = list()
        self.f1 = list()
        self.auc = list()
        self.l2 = list()

        self.preds = list()
        self.labels_hm = list()
        self.labels = list()
        self._init_streaming_metrics()

    def _init_streaming_metrics(self):
        self.thresholds = np.linspace(0, 0.02, 11)
        self.recall_sum = np.zeros_like(self.thresholds, dtype=np.float64)
        self.precision_sum = np.zeros_like(self.thresholds, dtype=np.float64)
        self.valid_frames = 0
        self.auc_sum = 0.0
        self.l2_sum = 0.0

    def _get_tracked_idx(self, labels_flat):
        if self.dataset in ["Holoassistgaze", "holoassistgaze", "Egoexo4dgaze", "egoexo4dgaze"]:
            tracked_idx = torch.where(labels_flat[:, 2] >= 0.5)[0]
        else:
            fixation_idx = 1 if self.dataset == "egteagaze" else 0
            tracked_idx = torch.where(labels_flat[:, 2] == fixation_idx)[0]
        return tracked_idx

    def _apply_edge_filter(self, labels_flat, tracked_idx):
        if self.edge_threshold <= 0.0:
            return tracked_idx
        labels_tracked = labels_flat.index_select(0, tracked_idx)
        edge_valid_idx = metrics.filter_edge_frames(labels_tracked, self.edge_threshold)
        return tracked_idx[edge_valid_idx]

    def _update_streaming_metrics(self, preds, labels_hm, labels):
        # preds: (B, 1, T, H, W), labels_hm: (B, T, H, W)
        labels_flat = labels.view(labels.size(0) * labels.size(1), labels.size(2))
        tracked_idx = self._get_tracked_idx(labels_flat)
        tracked_idx = self._apply_edge_filter(labels_flat, tracked_idx)
        if tracked_idx.numel() == 0:
            return

        preds_sw = preds.squeeze(1)
        Hl, Wl = labels_hm.size(-2), labels_hm.size(-1)
        Hp, Wp = preds_sw.size(-2), preds_sw.size(-1)
        if (Hp, Wp) != (Hl, Wl):
            preds_sw = F.interpolate(
                preds_sw.view(-1, 1, Hp, Wp),
                size=(Hl, Wl),
                mode="bilinear",
                align_corners=False,
            ).view(preds_sw.size(0), preds_sw.size(1), Hl, Wl)

        binary_labels = (labels_hm > 0.001).int()
        self.valid_frames += int(tracked_idx.numel())

        # Streaming F1 across thresholds (match adaptive_f1: mean over frames)
        binary_labels_bt = binary_labels.sum(dim=(2, 3)).view(-1)
        for i, thr in enumerate(self.thresholds):
            binary_preds = (preds_sw > float(thr)).int()
            tp_bt = (binary_preds * binary_labels).sum(dim=(2, 3)).view(-1)
            fg_preds_bt = binary_preds.sum(dim=(2, 3)).view(-1)

            tp = tp_bt.index_select(0, tracked_idx)
            fg_labels = binary_labels_bt.index_select(0, tracked_idx)
            fg_preds = fg_preds_bt.index_select(0, tracked_idx)

            recall_frame = tp / (fg_labels + 1e-6)
            precision_frame = tp / (fg_preds + 1e-6)

            self.recall_sum[i] += float(recall_frame.sum().item())
            self.precision_sum[i] += float(precision_frame.sum().item())

        # Streaming AUC/L2 (exact averages via weighted sum)
        auc = metrics.auc(preds, labels_hm, labels, dataset=self.dataset, edge_threshold=self.edge_threshold)
        l2 = metrics.l2_distance(
            preds,
            labels_hm,
            labels,
            dataset=self.dataset,
            edge_threshold=self.edge_threshold,
            l2_mode=self.l2_mode,
        )
        self.auc_sum += float(auc) * float(tracked_idx.numel())
        self.l2_sum += float(l2) * float(tracked_idx.numel())

    def update_stats(self, f1, recall, precision, auc, l2, preds, labels_hm, labels):
        labels_flat = labels.view(labels.size(0) * labels.size(1), -1)
        if self.dataset in ["Holoassistgaze", "holoassistgaze", "Egoexo4dgaze", "egoexo4dgaze"]:
            mb_size = torch.where(labels_flat[:, 2] >= 0.5)[0].size(0)
        else:
            fixation_idx = 1 if self.dataset == "egteagaze" else 0
            mb_size = torch.where(labels_flat[:, 2] == fixation_idx)[0].size(0)

        if mb_size == 0:
            return

        self.num_samples += mb_size
        self.f1.append(f1 * mb_size)
        self.recall.append(recall * mb_size)
        self.precision.append(precision * mb_size)
        self.auc.append(auc * mb_size)
        self.l2.append(l2 * mb_size)

        if self.store_heatmaps:
            self.preds.append(preds)
            self.labels_hm.append(labels_hm)
            self.labels.append(labels)
        else:
            self._update_streaming_metrics(preds, labels_hm, labels)

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        # Disabled per-iteration logging to reduce clutter when using tqdm progress bar
        # eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        # eta = str(datetime.timedelta(seconds=int(eta_sec)))
        # stats = {
        #     "split": "test_iter",
        #     "cur_iter": "{}/{}".format(cur_iter + 1, self.overall_iters),
        #     "eta": eta,
        #     "time_diff": self.iter_timer.seconds(),
        # }
        # logging.log_json_stats(stats)
        pass

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, efficiency_metrics=None, sigma=None):
        if self.store_heatmaps:
            preds = torch.cat(self.preds, dim=0)
            labels_hm = torch.cat(self.labels_hm, dim=0)
            labels = torch.cat(self.labels, dim=0)
            f1, recall, precision, threshold = metrics.adaptive_f1(
                preds, labels_hm, labels, dataset=self.dataset, edge_threshold=self.edge_threshold
            )
            auc = metrics.auc(preds, labels_hm, labels, dataset=self.dataset, edge_threshold=self.edge_threshold)
            l2 = metrics.l2_distance(
                preds,
                labels_hm,
                labels,
                dataset=self.dataset,
                edge_threshold=self.edge_threshold,
                l2_mode=self.l2_mode,
            )
        else:
            if self.valid_frames > 0:
                recall = self.recall_sum / float(self.valid_frames)
                precision = self.precision_sum / float(self.valid_frames)
            else:
                recall = np.zeros_like(self.thresholds, dtype=np.float64)
                precision = np.zeros_like(self.thresholds, dtype=np.float64)
            f1_all = (2 * recall * precision) / (recall + precision + 1e-6)
            max_idx = int(np.argmax(f1_all))
            f1 = float(f1_all[max_idx])
            recall = float(recall[max_idx])
            precision = float(precision[max_idx])
            threshold = float(self.thresholds[max_idx])
            if self.valid_frames > 0:
                auc = float(self.auc_sum / self.valid_frames)
                l2 = float(self.l2_sum / self.valid_frames)
            else:
                auc = 0.0
                l2 = 0.0

        self.stats = {
            "split": "test_final",
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "auc": auc,
            "l2": l2,
            "threshold": threshold
        }

        # Add sigma if provided
        if sigma is not None:
            self.stats["sigma"] = sigma

        # Add efficiency metrics if provided
        if efficiency_metrics is not None:
            self.stats.update(efficiency_metrics)

        logging.log_json_stats(self.stats)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1))
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class TrainGazeMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
        self.auc = ScalarMeter(cfg.LOG_PERIOD)
        self.auc_total = 0.0
        self.l2 = ScalarMeter(cfg.LOG_PERIOD)
        self.l2_total = 0.0
        self.entropy = ScalarMeter(cfg.LOG_PERIOD)
        self.entropy_total = 0.0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR
        self.threshold = -1

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.f1.reset()
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
        self.auc.reset()
        self.auc_total = 0.0
        self.l2.reset()
        self.l2_total = 0.0
        self.entropy.reset()
        self.entropy_total = 0.0
        self.num_samples = 0
        self.threshold = -1

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, f1, recall, precision, auc, l2, threshold, loss, entropy, lr, mb_size):
        """
        Update the current stats.
        Args:
            iou (float): iou metric.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.lr = lr
        self.loss.add_value(loss)
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        # Current minibatch stats
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
        self.auc.add_value(auc)
        self.auc_total += auc * mb_size
        self.l2.add_value(l2)
        self.l2_total += l2 * mb_size
        self.entropy.add_value(entropy)
        self.entropy_total += entropy * mb_size
        self.threshold = threshold

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1))
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "f1": self.f1.get_win_median(),
            "recall": self.recall.get_win_median(),
            "precision": self.precision.get_win_median(),
            "auc": self.auc.get_win_median(),
            "l2": self.l2.get_win_median(),
            "entropy": self.entropy.get_win_median(),
            "threshold": self.threshold
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            "loss": self.loss_total / self.num_samples,
            # "iou": self.iou_total / self.num_samples,
            "auc": self.auc_total / self.num_samples,
            "l2": self.l2_total / self.num_samples,
            "entropy": self.entropy_total / self.num_samples
        }
        recall = self.recall_total / self.num_samples
        precision = self.precision_total / self.num_samples
        f1 = 2 * recall * precision / (recall + precision + 1e-6)
        stats['f1'] = f1
        stats['recall'] = recall
        stats['precision'] = precision

        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err

            all_preds = torch.cat(self.all_preds, dim=0)
            all_labels = torch.cat(self.all_labels, dim=0)
            mean_cls_acc, recall_percls = metrics.mean_class_accuracy(preds=all_preds, labels=all_labels)
            stats["mean_clas_acc"] = mean_cls_acc
            stats["confusion_matrix"] = metrics.conf_matrix(preds=all_preds, labels=all_labels).tolist()

        logging.log_json_stats(stats)


class ValGazeMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.f1 = ScalarMeter(cfg.LOG_PERIOD)
        self.recall = ScalarMeter(cfg.LOG_PERIOD)
        self.recall_total = 0.0
        self.precision = ScalarMeter(cfg.LOG_PERIOD)
        self.precision_total = 0.0
        self.auc = ScalarMeter(cfg.LOG_PERIOD)
        self.auc_total = 0.0
        self.l2 = ScalarMeter(cfg.LOG_PERIOD)
        self.l2_total = 0.0
        self.threshold = -1
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.f1.reset()
        self.recall.reset()
        self.recall_total = 0.0
        self.precision.reset()
        self.precision_total = 0.0
        self.auc.reset()
        self.auc_total = 0.0
        self.l2.reset()
        self.l2_total = 0.0
        self.threshold = -1
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, f1, recall, precision, auc, l2, labels, threshold):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        labels_flat = labels.view(labels.size(0) * labels.size(1), -1)
        if self._cfg.TRAIN.DATASET in ["Holoassistgaze", "holoassistgaze", "Egoexo4dgaze", "egoexo4dgaze"]:
            mb_size = torch.where(labels_flat[:, 2] >= 0.5)[0].size(0)
        else:
            fixation_idx = 1 if self._cfg.TRAIN.DATASET == "egteagaze" else 0
            mb_size = torch.where(labels_flat[:, 2] == fixation_idx)[0].size(0)

        if mb_size == 0:
            return

        self.num_samples += mb_size
        self.f1.add_value(f1)
        self.recall.add_value(recall)
        self.recall_total += recall * mb_size
        self.precision.add_value(precision)
        self.precision_total += precision * mb_size
        self.auc.add_value(auc)
        self.auc_total += auc * mb_size
        self.l2.add_value(l2)
        self.l2_total += l2 * mb_size
        self.threshold = threshold

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "f1": self.f1.get_win_median(),
            "recall": self.recall.get_win_median(),
            "precision": self.precision.get_win_median(),
            "auc": self.auc.get_win_median(),
            "l2": self.l2.get_win_median(),
            "threshold": self.threshold
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        if self.num_samples == 0:
            stats = {
                "_type": "val_epoch",
                "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
                "time_diff": self.iter_timer.seconds(),
                "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
                "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
                "auc": 0.0,
                "l2": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
            }
            logging.log_json_stats(stats)
            return

        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
            # "iou": self.iou_total / self.num_samples,
            "auc": self.auc_total / self.num_samples,
            "l2": self.l2_total / self.num_samples
        }
        recall = self.recall_total / self.num_samples
        precision = self.precision_total / self.num_samples
        f1 = 2 * recall * precision / (recall + precision + 1e-6)
        stats['recall'] = recall
        stats['precision'] = precision
        stats['f1'] = f1

        logging.log_json_stats(stats)


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap


class EpochTimer:
    """
    A timer which computes the epoch time.
    """

    def __init__(self) -> None:
        self.timer = Timer()
        self.timer.reset()
        self.epoch_times = []

    def reset(self) -> None:
        """
        Reset the epoch timer.
        """
        self.timer.reset()
        self.epoch_times = []

    def epoch_tic(self):
        """
        Start to record time.
        """
        self.timer.reset()

    def epoch_toc(self):
        """
        Stop to record time.
        """
        self.timer.pause()
        self.epoch_times.append(self.timer.seconds())

    def last_epoch_time(self):
        """
        Get the time for the last epoch.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return self.epoch_times[-1]

    def avg_epoch_time(self):
        """
        Calculate the average epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.mean(self.epoch_times)

    def median_epoch_time(self):
        """
        Calculate the median epoch time among the recorded epochs.
        """
        assert len(self.epoch_times) > 0, "No epoch time has been recorded!"

        return np.median(self.epoch_times)
