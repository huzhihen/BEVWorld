"""
mmdet3d_plugin：MapTR / BEVFusionMapTR 训练插件。

BEVFusionMapTR 仅需 maptr 子模块（MapTRv2Head 等），勿在包初始化时导入
bevformer.detectors 或 VoVNet，以免与 mmdet3d 注册表冲突。
"""

from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder, MapTRNMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
    PhotoMetricDistortionMultiViewImage,
    PadMultiViewImage,
    NormalizeMultiviewImage,
    CustomCollect3D,
)
from .models.utils import *
from .models.opt.adamw import AdamW2
from .maptr import *
from .bevfusion_maptr import *
