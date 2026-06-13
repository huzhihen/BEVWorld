#!/usr/bin/env python3
"""
BEVFusion 测试 / 评估入口（OpenMMLab / mmdet3d 风格）。

用法::

    bash tools/dist_test.sh CONFIG CHECKPOINT 8 --eval bbox

仍兼容旧版 YAML 配置。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import mmcv
import torch
from mmcv import DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


def parse_args():
    parser = argparse.ArgumentParser(description="Test / eval a BEVFusion model")
    parser.add_argument("config", help="test config (.py or .yaml)")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result pickle path")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="fuse conv and bn for slightly faster inference",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="format results without evaluation",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='metrics, e.g. "bbox"',
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-dir", help="directory to save visualizations")
    parser.add_argument("--gpu-collect", action="store_true")
    parser.add_argument("--tmpdir", help="tmpdir for multi-gpu result collection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config key=value pairs",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="deprecated alias of --eval-options",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="kwargs passed to dataset.evaluate()",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError("Cannot specify both --options and --eval-options")
    if args.options:
        warnings.warn("--options is deprecated; use --eval-options")
        args.eval_options = args.options

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        'Specify at least one of "--out", "--eval", "--format-only", "--show", "--show-dir"'
    )
    if args.eval and args.format_only:
        raise ValueError("--eval and --format-only cannot be used together")
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a .pkl file")

    cfg = load_config(args.config, args.cfg_options)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.get("dist_params", dict(backend="nccl")))
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()
