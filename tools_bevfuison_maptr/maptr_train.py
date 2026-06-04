"""
BEVFusionMapTR 分布式训练入口（torchpack）。

用法（BEVWorld 根目录）::

    python tools_bevfuison_maptr/maptr_train.py \\
        projects/configs/bevfusion_maptr/bevfusion_maptr_nusc_r50_24ep_w_centerline_fusion.py \\
        --run-dir work_dirs/bevfusion_maptr_nusc
"""

import argparse
import importlib
import os
from pprint import pformat
import random
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from mmcv import Config, DictAction
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir


def import_plugin(cfg, config_path):
    """Load mmdet3d_plugin before mmdet3d.models to avoid registry conflicts."""
    if not getattr(cfg, "plugin", False):
        return

    if hasattr(cfg, "plugin_dir"):
        parts = cfg.plugin_dir.strip("/").split("/")
        module_path = ".".join(parts)
    else:
        module_dir = os.path.dirname(config_path)
        module_path = module_dir.replace("/", ".")
    importlib.import_module(module_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="python config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument("--no-validate", action="store_true", help="disable validation")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config entries, e.g. key=value key2=a,b",
    )
    return parser.parse_args()


def build_trainable_model(cfg):
    from mmdet3d.models.builder import build_model as build_fusion_model

    return build_fusion_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )


def _safe_config_text(cfg):
    try:
        return cfg.pretty_text
    except TypeError as exc:
        # mmcv<=1.x may call yapf.FormatCode(..., verify=True), but newer
        # yapf versions removed that kwarg. Fall back to a plain text dump so
        # training can proceed.
        if "verify" not in str(exc):
            raise
        cfg_dict = cfg._cfg_dict.to_dict() if hasattr(cfg._cfg_dict, "to_dict") else dict(cfg._cfg_dict)
        return pformat(cfg_dict, width=120, sort_dicts=False)


def _safe_dump_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(_safe_config_text(cfg))


def main():
    args = parse_args()
    dist.init()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin(cfg, args.config)

    from mmdet3d.apis import train_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import convert_sync_batchnorm, get_root_logger

    cfg.setdefault("seed", None)
    cfg.setdefault("deterministic", False)

    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir
    cfg.work_dir = args.run_dir

    _safe_dump_config(cfg, os.path.join(cfg.run_dir, os.path.basename(args.config)))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"Config:\n{_safe_config_text(cfg)}")

    if cfg.get("seed", None) is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, deterministic mode: {cfg.get('deterministic', False)}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.get("deterministic", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_trainable_model(cfg)
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
        distributed=True,
        validate=not args.no_validate,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
