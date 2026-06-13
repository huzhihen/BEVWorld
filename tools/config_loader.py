"""
BEVFusion 训练/测试配置加载工具。

支持两种配置格式：
  1. ``.py``  — OpenMMLab 风格（``mmcv.Config.fromfile``，推荐）
  2. ``.yaml/.yml`` — 旧版 torchpack 目录继承（向后兼容）

``train_model`` 使用 ``cfg.run_dir``，本模块会将 ``work_dir`` 与 ``run_dir`` 对齐。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from mmcv import Config


def load_config(config_path: str, cfg_options: Optional[Dict[str, Any]] = None) -> Config:
    """加载 BEVFusion 配置（自动识别 py / yaml）。"""
    config_path = os.path.abspath(config_path)
    ext = os.path.splitext(config_path)[1].lower()

    if ext in (".yaml", ".yml"):
        cfg = _load_yaml_config(config_path)
    elif ext == ".py":
        cfg = Config.fromfile(config_path)
    else:
        raise ValueError(f"Unsupported config extension: {ext} (use .py or .yaml)")

    if cfg_options:
        cfg.merge_from_dict(cfg_options)

    _normalize_runtime_fields(cfg, config_path)
    return cfg


def _load_yaml_config(config_path: str) -> Config:
    """旧版 torchpack YAML 配置（递归 merge default.yaml + recursive_eval）。"""
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval

    configs.load(config_path, recursive=True)
    return Config(recursive_eval(configs), filename=config_path)


def _normalize_runtime_fields(cfg: Config, config_path: str) -> None:
    """统一 work_dir / run_dir，并补齐 mmdet 常用默认值。"""
    default_work_dir = os.path.join(
        "./work_dirs",
        os.path.splitext(os.path.basename(config_path))[0],
    )

    work_dir = cfg.get("work_dir") or cfg.get("run_dir") or default_work_dir
    cfg.work_dir = work_dir
    cfg.run_dir = cfg.get("run_dir") or work_dir

    cfg.setdefault("dist_params", dict(backend="nccl"))
    cfg.setdefault("log_level", "INFO")
    cfg.setdefault("workflow", [("train", 1)])

    if cfg.get("seed") is None and "seed" not in cfg:
        cfg.seed = None
    cfg.setdefault("deterministic", False)
