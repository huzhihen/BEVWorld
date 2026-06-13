"""
BEVFusion nuScenes LiDAR-only TransFusion 配置（voxel 0.075）。

=============================================================================
使用说明
=============================================================================

数据预处理（标准 BEVFusion infos，非 MapTR pkl）::

    python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes

训练::

    bash tools/dist_train.sh \\
      projects/configs/bevfusion/bevfusion_nusc_lidar_transfusion_voxel0075.py 8 \\
      --work-dir work_dirs/bevfusion_lidar_v0075

评估::

    bash tools/dist_test.sh \\
      projects/configs/bevfusion/bevfusion_nusc_lidar_transfusion_voxel0075.py \\
      work_dirs/bevfusion_lidar_v0075/latest.pth 8 --eval bbox

对应旧版 YAML:
  configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml

Camera+LiDAR 融合配置见:
  projects/configs/bevfusion/bevfusion_nusc_camera_lidar_transfusion_swint_voxel0075.py

其他检测头:
  AnchorHead LiDAR:   bevfusion_nusc_lidar_anchor_voxel0075.py
  AnchorHead C+L:     bevfusion_nusc_camera_lidar_anchor_swint_voxel0075.py
  CenterHead Camera:  bevfusion_nusc_camera_centerhead_swint.py
  CenterHead C+L:     bevfusion_nusc_camera_lidar_centerhead_swint_voxel0075.py
=============================================================================
"""

_base_ = ["./_base_/default_runtime.py"]

dataset_type = "NuScenesDataset"
dataset_root = "data/nuscenes/"
data_root = dataset_root

load_dim = 5
use_dim = 5
reduce_beams = 32
load_augmented = None
sweeps_num = 9
gt_paste_stop_epoch = 15

voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
sparse_shape = [1440, 1440, 41]
grid_size = [1440, 1440, 41]

object_classes = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)

augment3d = dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5,
)

model = dict(
    type="BEVFusion",
    encoders=dict(
        camera=None,
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=[120000, 160000],
            ),
            backbone=dict(
                type="SparseEncoder",
                in_channels=5,
                sparse_shape=sparse_shape,
                output_channels=128,
                order=("conv", "norm", "act"),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
                block_type="basicblock",
            ),
        ),
    ),
    fuser=None,
    decoder=dict(
        backbone=dict(
            type="SECOND",
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type="BN", eps=1.0e-3, momentum=0.01),
            conv_cfg=dict(type="Conv2d", bias=False),
        ),
        neck=dict(
            type="SECONDFPN",
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type="BN", eps=1.0e-3, momentum=0.01),
            upsample_cfg=dict(type="deconv", bias=False),
            use_conv_for_no_stride=True,
        ),
    ),
    heads=dict(
        map=None,
        object=dict(
            type="TransFusionHead",
            num_proposals=200,
            auxiliary=True,
            in_channels=512,
            hidden_channel=128,
            num_classes=10,
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation="relu",
            common_heads=dict(
                center=[2, 2],
                height=[1, 2],
                dim=[3, 2],
                rot=[2, 2],
                vel=[2, 2],
            ),
            bbox_coder=dict(
                type="TransFusionBBoxCoder",
                pc_range=point_cloud_range[:2],
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=10,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction="mean",
                loss_weight=1.0,
            ),
            loss_heatmap=dict(
                type="GaussianFocalLoss",
                reduction="mean",
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
            train_cfg=dict(
                dataset="nuScenes",
                point_cloud_range=point_cloud_range,
                grid_size=grid_size,
                voxel_size=voxel_size,
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                assigner=dict(
                    type="HungarianAssigner3D",
                    iou_calculator=dict(type="BboxOverlaps3D", coordinate="lidar"),
                    cls_cost=dict(type="FocalLossCost", gamma=2.0, alpha=0.25, weight=0.15),
                    reg_cost=dict(type="BBoxBEVL1Cost", weight=0.25),
                    iou_cost=dict(type="IoU3DCost", weight=0.25),
                ),
            ),
            test_cfg=dict(
                dataset="nuScenes",
                grid_size=grid_size,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2],
                nms_type=None,
            ),
        ),
    ),
)

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="ObjectPaste",
        stop_epoch=gt_paste_stop_epoch,
        db_sampler=dict(
            dataset_root=dataset_root,
            info_path=dataset_root + "nuscenes_dbinfos_train.pkl",
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5,
                ),
            ),
            classes=object_classes,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2,
            ),
            points_loader=dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=load_dim,
                use_dim=use_dim,
                reduce_beams=reduce_beams,
            ),
        ),
    ),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=augment3d["scale"],
        rot_lim=augment3d["rotate"],
        trans_lim=augment3d["translate"],
        is_train=True,
    ),
    dict(type="RandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=object_classes),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(
        type="Collect3D",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "lidar2ego",
            "lidar_aug_matrix",
        ],
    ),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="CBGSDataset",
        dataset=dict(
            type=dataset_type,
            dataset_root=dataset_root,
            ann_file=dataset_root + "nuscenes_infos_train.pkl",
            pipeline=train_pipeline,
            object_classes=object_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        test_mode=False,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        dataset_root=dataset_root,
        ann_file=dataset_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)

optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="cyclic",
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy="cyclic",
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

max_epochs = 20
runner = dict(type="CustomEpochBasedRunner", max_epochs=max_epochs)
evaluation = dict(interval=1, pipeline=test_pipeline)
