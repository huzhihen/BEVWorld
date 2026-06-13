"""
BEVFusion nuScenes Camera-only CenterHead 配置（Swin-T + LSSFPN）。

训练::

    bash tools/dist_train.sh \\
      projects/configs/bevfusion/bevfusion_nusc_camera_centerhead_swint.py 8 \\
      --work-dir work_dirs/bevfusion_camera_centerhead \\
      --cfg-options model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth

评估::

    bash tools/dist_test.sh \\
      projects/configs/bevfusion/bevfusion_nusc_camera_centerhead_swint.py \\
      work_dirs/bevfusion_camera_centerhead/latest.pth 8 --eval bbox

对应旧版 YAML:
  configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml
"""

_base_ = ["./_base_/default_runtime.py"]

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

centerhead_tasks = [
    ["car"],
    ["truck", "construction_vehicle"],
    ["bus", "trailer"],
    ["barrier"],
    ["motorcycle", "bicycle"],
    ["pedestrian", "traffic_cone"],
]

dataset_type = "NuScenesDataset"
dataset_root = "data/nuscenes/"
image_size = [256, 704]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
grid_size = [1024, 1024, 1]

load_dim = 5
use_dim = 5
reduce_beams = 32
load_augmented = None
sweeps_num = 9
gt_paste_stop_epoch = 15

augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True),
)
augment3d = dict(
    scale=[0.95, 1.05],
    rotate=[-0.3925, 0.3925],
    translate=0.0,
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

model = dict(
    type="BEVFusion",
    encoders=dict(
        camera=dict(
            backbone=dict(
                type="SwinTransformer",
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint="pretrained/swint-nuimages-pretrained.pth",
                ),
            ),
            neck=dict(
                type="GeneralizedLSSFPN",
                in_channels=[192, 384, 768],
                out_channels=256,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type="BN2d", requires_grad=True),
                act_cfg=dict(type="ReLU", inplace=True),
                upsample_cfg=dict(mode="bilinear", align_corners=False),
            ),
            vtransform=dict(
                type="DepthLSSTransform",
                in_channels=256,
                out_channels=80,
                image_size=image_size,
                feature_size=[image_size[0] // 8, image_size[1] // 8],
                xbound=[-51.2, 51.2, 0.4],
                ybound=[-51.2, 51.2, 0.4],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
            ),
        ),
        lidar=None,
    ),
    fuser=None,
    decoder=dict(
        backbone=dict(
            type="GeneralizedResNet",
            in_channels=80,
            blocks=[[2, 128, 2], [2, 256, 2], [2, 512, 1]],
        ),
        neck=dict(
            type="LSSFPN",
            in_indices=[-1, 0],
            in_channels=[512, 128],
            out_channels=256,
            scale_factor=2,
        ),
    ),
    heads=dict(
        map=None,
        object=dict(
            type="CenterHead",
            in_channels=256,
            tasks=centerhead_tasks,
            common_heads=dict(reg=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
            share_conv_channel=64,
            bbox_coder=dict(
                type="CenterPointBBoxCoder",
                pc_range=point_cloud_range,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=9,
            ),
            separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
            loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
            norm_bbox=True,
            train_cfg=dict(
                point_cloud_range=point_cloud_range,
                grid_size=grid_size,
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            test_cfg=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type="rotate",
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2,
            ),
        ),
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
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
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d["resize"][0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d["rotate"],
        rand_flip=True,
        is_train=True,
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
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="GridMask",
        use_h=True,
        use_w=True,
        max_epoch=20,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=augment2d["gridmask"]["prob"],
        fixed_prob=augment2d["gridmask"]["fixed_prob"],
    ),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(
        type="Collect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "lidar2image",
            "camera2lidar",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ],
    ),
    dict(type="GTDepth", keyframe_only=True),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
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
        type="ImageAug3D",
        final_dim=image_size,
        resize_lim=augment2d["resize"][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False,
    ),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(
        type="Collect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=[
            "camera_intrinsics",
            "camera2ego",
            "lidar2ego",
            "lidar2camera",
            "lidar2image",
            "camera2lidar",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ],
    ),
    dict(type="GTDepth", keyframe_only=True),
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

optimizer = dict(
    type="AdamW",
    lr=2.0e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0),
            relative_position_bias_table=dict(decay_mult=0),
            **{"encoders.camera.backbone": dict(lr_mult=0.1)},
        ),
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="cyclic",
    target_ratio=5.0,
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy="cyclic",
    cyclic_times=1,
    step_ratio_up=0.4,
)

max_epochs = 20
runner = dict(type="CustomEpochBasedRunner", max_epochs=max_epochs)
evaluation = dict(interval=1, pipeline=test_pipeline)
