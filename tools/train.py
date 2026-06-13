#!/usr/bin/env python3
"""
BEVFusion 训练入口（OpenMMLab / mmdet3d 风格）。

=============================================================================
使用说明
=============================================================================

单卡::

    python tools/train.py projects/configs/bevfusion/bevfusion_nusc_lidar_transfusion_voxel0075.py

多卡（推荐）::

    bash tools/dist_train.sh \
      projects/configs/bevfusion/bevfusion_nusc_lidar_transfusion_voxel0075.py 8 \
      --work-dir work_dirs/bevfusion_lidar_v0075

仍兼容旧版 YAML（torchpack 目录继承）::

    bash tools/dist_train.sh configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml 8

覆盖配置项::

    bash tools/dist_train.sh CONFIG 8 --cfg-options data.samples_per_gpu=2

=============================================================================
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import convert_sync_batchnorm, get_root_logger
from mmdet.apis import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BEVFusion model")
    parser.add_argument("config", help="train config (.py or .yaml)")
    parser.add_argument("--work-dir", help="directory to save logs and checkpoints")
    parser.add_argument(
        "--resume-from", help="checkpoint file to resume from",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="disable validation during training",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="set CUDNN deterministic options",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help='override config, e.g. data.samples_per_gpu=2',
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="deprecated alias of --cfg-options",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError("Cannot specify both --options and --cfg-options")
    if args.options:
        warnings.warn("--options is deprecated; use --cfg-options")
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = load_config(args.config, args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
        cfg.run_dir = args.work_dir
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from

    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.get("dist_params", dict(backend="nccl")))
    else:
        # train_model 内部固定使用 DDP，单卡时也需初始化单进程分布式
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_dist("pytorch", **cfg.get("dist_params", dict(backend="nccl")))
        distributed = True
    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.get("log_level", "INFO"))

    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    if cfg.get("seed") is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, deterministic: {cfg.get('deterministic', False)}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        set_random_seed(cfg.seed, deterministic=cfg.get("deterministic", False))
        if cfg.get("deterministic", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()

    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
