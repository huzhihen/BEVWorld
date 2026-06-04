"""
BEVFusionMapTR 联合训练数据集。

在 CustomNuScenesOfflineLocalMapDataset 基础上注册为独立类型，
用于读取 tools_bevfuison_maptr 生成的联合 pkl（3D 检测 GT + 矢量地图 GT + 时序信息）。

配置示例::

    dataset_type = 'BEVFusionMapTRJointDataset'
    ann_file = 'data/nuscenes/nuscenes_bevfusion_maptr_map_infos_temporal_train.pkl'
"""

from mmdet.datasets import DATASETS

from projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset import (
    CustomNuScenesOfflineLocalMapDataset,
)


@DATASETS.register_module()
class BEVFusionMapTRJointDataset(CustomNuScenesOfflineLocalMapDataset):
    """BEVFusion + MapTRv2 联合监督数据集。

    与 :class:`CustomNuScenesOfflineLocalMapDataset` 行为一致，默认开启地图 GT，
    并与 LoadAnnotations3D 输出的 3D 检测 GT 共存（gt_map_* / gt_bboxes_3d 分离）。
    """

    def __init__(self, with_map_gt=True, *args, **kwargs):
        super().__init__(with_map_gt=with_map_gt, *args, **kwargs)
