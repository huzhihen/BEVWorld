"""
BEVFusion 推理速度 benchmark。

用法::

    python tools/benchmark.py CONFIG CHECKPOINT [--samples 2000] [--fp16]

支持 ``.py`` 与旧版 ``.yaml`` 配置。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config (.py or .yaml)")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--samples", default=2000, type=int, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, type=int, help="interval of logging")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    num_warmup = 5
    pure_inf_time = 0

    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f"Done image [{i + 1:<3}/ {args.samples}], fps: {fps:.1f} img / s")

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f"Overall fps: {fps:.1f} img / s")
            break


if __name__ == "__main__":
    main()
