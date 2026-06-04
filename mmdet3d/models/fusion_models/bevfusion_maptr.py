"""BEVFusion + MapTRv2 joint multi-task model.

This module extends the original BEVFusion model to support a ``vectormap``
head slot that routes multi-camera image features and lidar BEV features to a
MapTRv2Head for vectorized HD-map prediction.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Camera Encoder  (backbone → neck → vtransform)      │
    │    ├─ multi-level image feats (for vectormap head)   │
    │    └─ camera BEV feature (for fuser)                 │
    │                                                      │
    │  LiDAR Encoder   (voxelize → sparse backbone)        │
    │    └─ lidar BEV feature (for fuser + optional map)   │
    │                                                      │
    │  Fuser  (camera BEV + lidar BEV → fused BEV)        │
    │                                                      │
    │  Decoder  (backbone + neck)                          │
    │    └─ decoded features (for object head)             │
    │                                                      │
    │  Heads                                               │
    │    ├─ object:    detection head                      │
    │    └─ vectormap: MapTRv2Head (vectorized)            │
    └──────────────────────────────────────────────────────┘
"""

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.batchnorm import _BatchNorm

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.models import FUSIONMODELS

from .bevfusion import BEVFusion

__all__ = ["BEVFusionMapTR"]


@FUSIONMODELS.register_module()
class BEVFusionMapTR(BEVFusion):
    """BEVFusion extended with a MapTRv2 vectorized-map head.

    Compared with the vanilla :class:`BEVFusion`, this class adds:

    *   A ``vectormap`` head slot that accepts a ``MapTRv2Head`` config.
    *   During camera feature extraction, multi-level image features are
        cached *before* the view-transform so that the vectormap head can
        consume ``[B, N, C, H, W]`` camera features directly.
    *   The lidar BEV feature (before fusion) can optionally be forwarded to
        the vectormap head when ``use_lidar_for_vectormap`` is enabled.

    Extra forward kwargs consumed by this model:

    *   ``gt_map_bboxes_3d`` – vectorized map GT (``LiDARInstanceLines``)
    *   ``gt_map_labels_3d`` – per-instance map class labels
    *   ``gt_seg_mask`` / ``gt_pv_seg_mask`` – auxiliary segmentation masks
    """

    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        # Pop BEVFusionMapTR-only options before passing to parent
        freeze_modules = kwargs.pop("freeze_modules", None)
        self.use_lidar_for_vectormap = kwargs.pop("use_lidar_for_vectormap", False)

        # Pop vectormap config before super().__init__ because the parent
        # class does not know how to build it via build_head (it is registered
        # under mmdet HEADS, not mmdet3d).
        heads = copy.deepcopy(heads)
        vectormap_cfg = heads.pop("vectormap", None)
        train_cfg = kwargs.get("train_cfg")
        test_cfg = kwargs.get("test_cfg")

        super().__init__(
            encoders=encoders,
            fuser=fuser,
            decoder=decoder,
            heads=heads,
            **kwargs,
        )

        # Build vectormap head separately ─ it lives in the plugin registry
        if vectormap_cfg is not None:
            from mmdet.models import build_head as mmdet_build_head

            vectormap_cfg = copy.deepcopy(vectormap_cfg)
            if train_cfg is not None and "train_cfg" not in vectormap_cfg:
                vectormap_cfg["train_cfg"] = train_cfg.get("pts", train_cfg)
            if test_cfg is not None and "test_cfg" not in vectormap_cfg:
                vectormap_cfg["test_cfg"] = test_cfg.get("pts", test_cfg)
            self.heads["vectormap"] = mmdet_build_head(vectormap_cfg)
            if "vectormap" not in self.loss_scale:
                self.loss_scale["vectormap"] = 1.0

        # Runtime cache: filled during extract_camera_features
        self._mlvl_img_feats: Optional[List[torch.Tensor]] = None

        # Freeze specified module groups (e.g. ["encoders.lidar"])
        self._freeze_module_patterns: list = freeze_modules or []
        if self._freeze_module_patterns:
            self._apply_freeze()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------
    def _apply_freeze(self) -> None:
        """Freeze parameters whose name contains any of the patterns."""
        frozen_count = 0
        for name, param in self.named_parameters():
            if any(p in name for p in self._freeze_module_patterns):
                param.requires_grad = False
                frozen_count += 1
        if frozen_count > 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(
                f"[BEVFusionMapTR] Frozen {frozen_count} parameters "
                f"matching patterns: {self._freeze_module_patterns}"
            )

    def train(self, mode: bool = True):
        """Override to keep BatchNorm layers in frozen modules in eval mode."""
        super().train(mode)
        if mode and self._freeze_module_patterns:
            for name, module in self.named_modules():
                if any(p in name for p in self._freeze_module_patterns):
                    if isinstance(module, (_BatchNorm, nn.SyncBatchNorm)):
                        module.eval()
        return self

    # ------------------------------------------------------------------
    # Camera feature extraction – override to cache multi-level feats
    # ------------------------------------------------------------------
    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        # ``x`` may be a tuple/list of multi-level features from the neck.
        # Cache them in [B, N, C_i, H_i, W_i] format for MapTRv2Head.
        if isinstance(x, (list, tuple)):
            mlvl = []
            for feat in x:
                BN, Ci, Hi, Wi = feat.size()
                mlvl.append(feat.view(B, int(BN / B), Ci, Hi, Wi))
            self._mlvl_img_feats = mlvl
            # BEVFusion vtransform expects a single tensor – take the first
            x = x[0]
        else:
            BN, Ci, Hi, Wi = x.size()
            self._mlvl_img_feats = [x.view(B, int(BN / B), Ci, Hi, Wi)]

        BN, Cx, Hx, Wx = x.size()
        x = x.view(B, int(BN / B), Cx, Hx, Wx)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            None,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        points=None,
        metas=None,
        lidar2ego=None,
        lidar_aug_matrix=None,
        camera2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        img=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        # ---- new: vector-map GT ----
        gt_map_bboxes_3d=None,
        gt_map_labels_3d=None,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
        gt_depth=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_map_bboxes_3d=gt_map_bboxes_3d,
                gt_map_labels_3d=gt_map_labels_3d,
                gt_seg_mask=gt_seg_mask,
                gt_pv_seg_mask=gt_pv_seg_mask,
                gt_depth=gt_depth,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego=None,
        lidar2ego=None,
        lidar2camera=None,
        lidar2image=None,
        camera_intrinsics=None,
        camera2lidar=None,
        img_aug_matrix=None,
        lidar_aug_matrix=None,
        metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_map_bboxes_3d=None,
        gt_map_labels_3d=None,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
        gt_depth=None,
        **kwargs,
    ):
        (
            img,
            metas,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
        ) = self._normalize_inputs(
            img,
            metas,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            kwargs,
        )

        # Reset cache
        self._mlvl_img_feats = None

        # ----- Sensor feature extraction -----
        features = []
        lidar_bev_feat = None
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_features(points, "lidar")
                lidar_bev_feat = feature  # cache for vectormap head
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            features = features[::-1]

        if self.fuser is not None:
            features = self._align_fuser_inputs(features)
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        # ----- Decoder (shared BEV backbone + neck for detection) -----
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        # ----- Head dispatch -----
        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif head_type == "vectormap":
                    losses = self._forward_vectormap_train(
                        head,
                        lidar_bev_feat,
                        metas,
                        gt_map_bboxes_3d,
                        gt_map_labels_3d,
                        gt_seg_mask=gt_seg_mask,
                        gt_pv_seg_mask=gt_pv_seg_mask,
                        gt_depth=gt_depth,
                    )
                else:
                    raise ValueError(f"unsupported head: {head_type}")

                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{head_type}/{name}"] = (
                            val * self.loss_scale[head_type]
                        )
                    else:
                        outputs[f"stats/{head_type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_type == "vectormap":
                    map_results = self._forward_vectormap_test(
                        head, lidar_bev_feat, metas
                    )
                    for k in range(batch_size):
                        outputs[k].update(map_results[k])
                else:
                    raise ValueError(f"unsupported head: {head_type}")
            return outputs

    # ------------------------------------------------------------------
    # Vectormap head helpers
    # ------------------------------------------------------------------
    def _forward_vectormap_train(
        self,
        head,
        lidar_feat,
        metas,
        gt_map_bboxes_3d,
        gt_map_labels_3d,
        gt_seg_mask=None,
        gt_pv_seg_mask=None,
        gt_depth=None,
    ) -> dict:
        """Run MapTRv2Head forward + loss during training.

        Args:
            head: The MapTRv2Head instance.
            lidar_feat: LiDAR BEV features (before fusion), or None.
            metas: Batch metadata list.
            gt_map_bboxes_3d: List of LiDARInstanceLines per sample.
            gt_map_labels_3d: List of label tensors per sample.
            gt_seg_mask: Optional BEV segmentation mask for aux loss.
            gt_pv_seg_mask: Optional PV segmentation mask for aux loss.

        Returns:
            dict of loss tensors.
        """
        mlvl_feats = self._mlvl_img_feats
        assert mlvl_feats is not None, (
            "Multi-level camera features not available. "
            "Ensure camera encoder is present."
        )

        # Prepare img_metas – BEVFusion stores them as a list of dicts
        # under the ``metas`` key; MapTRv2Head expects ``img_metas``.
        img_metas = self._prepare_vectormap_img_metas(metas)

        if not self.use_lidar_for_vectormap:
            lidar_feat = None

        # Forward through MapTRv2Head
        outs = head(mlvl_feats, lidar_feat, img_metas, prev_bev=None)

        depth = outs.pop("depth", None)
        losses = {}

        # Depth loss (if available)
        if depth is not None and hasattr(head, "transformer"):
            encoder = getattr(head.transformer, "encoder", None)
            if encoder is not None and hasattr(encoder, "get_depth_loss"):
                if gt_depth is not None:
                    if isinstance(gt_depth, (list, tuple)):
                        gt_depth = torch.stack(
                            [torch.as_tensor(each) for each in gt_depth], dim=0
                        )
                    elif not torch.is_tensor(gt_depth):
                        gt_depth = torch.as_tensor(gt_depth)
                    gt_depth = gt_depth.to(device=depth.device, dtype=torch.float32)
                    loss_depth = encoder.get_depth_loss(gt_depth, depth)
                    loss_depth = torch.nan_to_num(loss_depth)
                    losses.update(loss_depth=loss_depth)

        # Main loss
        loss_inputs = [
            gt_map_bboxes_3d,
            gt_map_labels_3d,
            gt_seg_mask,
            gt_pv_seg_mask,
            outs,
        ]
        losses.update(head.loss(*loss_inputs, img_metas=img_metas))

        # One-to-many auxiliary loss
        if hasattr(head, "k_one2many") and head.k_one2many > 0:
            k_one2many = head.k_one2many
            multi_gt_bboxes_3d = copy.deepcopy(gt_map_bboxes_3d)
            multi_gt_labels_3d = copy.deepcopy(gt_map_labels_3d)
            for idx, (each_gt_bboxes_3d, each_gt_labels_3d) in enumerate(
                zip(multi_gt_bboxes_3d, multi_gt_labels_3d)
            ):
                each_gt_bboxes_3d.instance_list = (
                    each_gt_bboxes_3d.instance_list * k_one2many
                )
                each_gt_bboxes_3d.instance_labels = (
                    each_gt_bboxes_3d.instance_labels * k_one2many
                )
                multi_gt_labels_3d[idx] = each_gt_labels_3d.repeat(k_one2many)

            one2many_outs = outs.get("one2many_outs", None)
            if one2many_outs is not None:
                loss_inputs_o2m = [
                    multi_gt_bboxes_3d,
                    multi_gt_labels_3d,
                    gt_seg_mask,
                    gt_pv_seg_mask,
                    one2many_outs,
                ]
                loss_dict_o2m = head.loss(*loss_inputs_o2m, img_metas=img_metas)
                lambda_one2many = getattr(head, "lambda_one2many", 1.0)
                for key, value in loss_dict_o2m.items():
                    losses[f"{key}_one2many"] = (
                        losses.get(f"{key}_one2many", 0) + value * lambda_one2many
                    )

        return losses

    def _forward_vectormap_test(
        self,
        head,
        lidar_feat,
        metas,
    ) -> list:
        """Run MapTRv2Head forward + decode during inference.

        Returns:
            List of dicts, one per sample, with keys like
            ``map_boxes_3d``, ``map_scores_3d``, ``map_labels_3d``,
            ``map_pts_3d``.
        """
        mlvl_feats = self._mlvl_img_feats
        assert mlvl_feats is not None

        img_metas = self._prepare_vectormap_img_metas(metas)
        if not self.use_lidar_for_vectormap:
            lidar_feat = None

        outs = head(mlvl_feats, lidar_feat, img_metas, prev_bev=None)
        bbox_list = head.get_bboxes(outs, img_metas, rescale=False)

        results = []
        for bboxes, scores, labels, pts in bbox_list:
            results.append(
                {
                    "map_boxes_3d": bboxes.to("cpu"),
                    "map_scores_3d": scores.cpu(),
                    "map_labels_3d": labels.cpu(),
                    "map_pts_3d": pts.to("cpu"),
                }
            )
        return results

    @classmethod
    def _prepare_vectormap_img_metas(cls, metas) -> list:
        img_metas = cls._prepare_img_metas(metas)
        isolated_metas = []
        for meta in img_metas:
            isolated_meta = dict(meta)
            if "lidar_aug_matrix" in isolated_meta:
                aug_matrix = isolated_meta["lidar_aug_matrix"]
                if isinstance(aug_matrix, torch.Tensor):
                    isolated_meta["lidar_aug_matrix"] = torch.eye(
                        4, dtype=aug_matrix.dtype, device=aug_matrix.device
                    )
                else:
                    dtype = getattr(np.asarray(aug_matrix), "dtype", np.float32)
                    isolated_meta["lidar_aug_matrix"] = np.eye(4, dtype=dtype)
            isolated_metas.append(isolated_meta)
        return isolated_metas

    @classmethod
    def _prepare_img_metas(cls, metas) -> list:
        """Convert BEVFusion ``metas`` format to a list of dicts.

        BEVFusion typically passes ``metas`` as a list of dicts (one per
        sample).  MapTRv2Head expects the same under the name ``img_metas``.
        The plugin map datasets wrap temporal history in a dict keyed by frame
        index even when ``queue_length == 1``; unwrap the latest frame so the
        downstream camera transform and MapTR head both see a plain per-sample
        metadata dict.
        """
        def _unwrap_single(meta):
            if (
                isinstance(meta, dict)
                and meta
                and all(isinstance(key, int) for key in meta.keys())
            ):
                return meta[max(meta.keys())]
            return meta

        if isinstance(metas, (list, tuple)):
            return [_unwrap_single(meta) for meta in metas]
        return [_unwrap_single(metas)]

    @staticmethod
    def _stack_meta_tensor(reference: torch.Tensor, img_metas: list, key: str):
        if not img_metas or key not in img_metas[0]:
            return None

        values = []
        for meta in img_metas:
            value = meta[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            values.append(value)
        return reference.new_tensor(np.asarray(values))

    @classmethod
    def _stack_meta_tensor_any(cls, reference: torch.Tensor, img_metas: list, keys):
        for key in keys:
            value = cls._stack_meta_tensor(reference, img_metas, key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _safe_inverse(matrix: torch.Tensor) -> torch.Tensor:
        orig_dtype = matrix.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            matrix = matrix.float()
        inv = torch.inverse(matrix)
        return inv.to(orig_dtype)

    @staticmethod
    def _align_fuser_inputs(features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) <= 1:
            return features

        target_hw = features[-1].shape[-2:]
        aligned = []
        for feat in features:
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(
                    feat,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feat)
        return aligned

    @staticmethod
    def _default_img_aug_matrix(img: torch.Tensor) -> torch.Tensor:
        batch_size, num_cams = img.shape[:2]
        eye = torch.eye(4, device=img.device, dtype=img.dtype)
        return eye.view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)

    @staticmethod
    def _default_lidar_aug_matrix(img: torch.Tensor) -> torch.Tensor:
        batch_size = img.shape[0]
        eye = torch.eye(4, device=img.device, dtype=img.dtype)
        return eye.view(1, 4, 4).repeat(batch_size, 1, 1)

    def _normalize_inputs(
        self,
        img,
        metas,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        kwargs,
    ):
        raw_img_metas = kwargs.get("img_metas")
        if metas is None and raw_img_metas is not None:
            metas = raw_img_metas
        metas = self._prepare_img_metas(metas)

        if img is not None and img.dim() == 6:
            if img.size(1) != 1:
                raise ValueError(
                    "BEVFusionMapTR only supports queue_length == 1 in the "
                    "current joint-training path."
                )
            img = img[:, -1]

        if img is None:
            return (
                img,
                metas,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
            )

        if camera2ego is None:
            camera2ego = self._stack_meta_tensor(img, metas, "camera2ego")
        if lidar2ego is None:
            lidar2ego = self._stack_meta_tensor(img, metas, "lidar2ego")
        if lidar2camera is None:
            lidar2camera = self._stack_meta_tensor_any(
                img, metas, ("lidar2camera", "lidar2cam")
            )
        if lidar2image is None:
            lidar2image = self._stack_meta_tensor_any(
                img, metas, ("lidar2image", "lidar2img")
            )
        if camera_intrinsics is None:
            camera_intrinsics = self._stack_meta_tensor_any(
                img, metas, ("camera_intrinsics", "cam_intrinsic")
            )
        if camera2lidar is None:
            camera2lidar = self._stack_meta_tensor_any(
                img, metas, ("camera2lidar", "cam2lidar")
            )
        if camera2lidar is None and lidar2camera is not None:
            camera2lidar = self._safe_inverse(lidar2camera)
        if lidar2camera is None and camera2lidar is not None:
            lidar2camera = self._safe_inverse(camera2lidar)
        if lidar2image is None and camera_intrinsics is not None and lidar2camera is not None:
            lidar2image = camera_intrinsics.matmul(lidar2camera)
        if img_aug_matrix is None:
            img_aug_matrix = self._stack_meta_tensor(img, metas, "img_aug_matrix")
        if lidar_aug_matrix is None:
            lidar_aug_matrix = self._stack_meta_tensor(img, metas, "lidar_aug_matrix")
        if img_aug_matrix is None:
            img_aug_matrix = self._default_img_aug_matrix(img)
        if lidar_aug_matrix is None:
            lidar_aug_matrix = self._default_lidar_aug_matrix(img)

        return (
            img,
            metas,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
        )
