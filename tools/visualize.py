"""
BEVFusion 可视化工具（GT / 预测）。

用法::

    # GT 可视化（单卡）
    python tools/visualize.py CONFIG --mode gt --split val --out-dir viz

    # 预测可视化（多卡）
    bash tools/dist_test.sh CONFIG CKPT 1 --eval bbox   # 或单独跑 visualize:
    python tools/visualize.py CONFIG --mode pred --checkpoint CKPT --launcher pytorch

支持 ``.py`` 与旧版 ``.yaml`` 配置。
"""

from __future__ import annotations

import argparse
import os
import sys

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_config
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="pytorch",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)

    distributed = args.launcher != "none"
    if distributed:
        init_dist(args.launcher, **cfg.get("dist_params", dict(backend="nccl")))
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_dist("pytorch", **cfg.get("dist_params", dict(backend="nccl")))
        distributed = True

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    model = None
    if args.mode == "pred":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode pred")
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    rank, _ = get_dist_info()
    if rank != 0:
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        name = "{}-{}".format(metas["timestamp"], metas["token"])

        outputs = None
        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and outputs is not None and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and outputs is not None and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                )

        if "points" in data:
            lidar = data["points"].data[0][0].numpy()
            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.get("map_classes", []),
            )


if __name__ == "__main__":
    main()
