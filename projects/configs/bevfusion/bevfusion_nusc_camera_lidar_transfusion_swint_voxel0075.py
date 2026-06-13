"""
BEVFusion nuScenes Camera + LiDAR 融合检测配置（Swin-T + TransFusion, voxel 0.075）。

=============================================================================
使用说明
=============================================================================

数据预处理（与 LiDAR-only 相同，标准 BEVFusion infos）::

    python tools/create_data.py nuscenes \\
      --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes

下载预训练权重::

    bash tools/download_pretrained.sh
    # 得到 pretrained/swint-nuimages-pretrained.pth

从零训练（Camera + LiDAR 联合）::

    bash tools/dist_train.sh \\
      projects/configs/bevfusion/bevfusion_nusc_camera_lidar_transfusion_swint_voxel0075.py 8 \\
      --work-dir work_dirs/bevfusion_camera_lidar_v0075 \\
      --cfg-options model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth

显存参考（DepthLSSTransform 较重）::

    - 12GB GPU: 保持默认 samples_per_gpu=1
    - 24GB GPU: --cfg-options data.samples_per_gpu=2
    - 原论文 8×A100 级别: samples_per_gpu=4

推荐：先训 LiDAR-only，再加载其权重微调融合模型（与原 README 一致）::

    bash tools/dist_train.sh \\
      projects/configs/bevfusion/bevfusion_nusc_camera_lidar_transfusion_swint_voxel0075.py 8 \\
      --work-dir work_dirs/bevfusion_camera_lidar_v0075 \\
      --cfg-options \\
        model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth \\
      --load-from work_dirs/bevfusion_lidar_v0075/latest.pth

评估::

    bash tools/dist_test.sh \\
      projects/configs/bevfusion/bevfusion_nusc_camera_lidar_transfusion_swint_voxel0075.py \\
      work_dirs/bevfusion_camera_lidar_v0075/latest.pth 8 --eval bbox

对应旧版 YAML:
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
=============================================================================
"""

_base_ = ["./bevfusion_nusc_lidar_transfusion_voxel0075.py"]

image_size = [256, 704]
augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(prob=0.0, fixed_prob=True),
)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

model = dict(
    encoders=dict(
        camera=dict(
            _delete_=True,
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
                xbound=[-54.0, 54.0, 0.3],
                ybound=[-54.0, 54.0, 0.3],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[1.0, 60.0, 0.5],
                downsample=2,
            ),
        ),
    ),
    fuser=dict(
        _delete_=True,
        type="ConvFuser",
        in_channels=[80, 256],
        out_channels=256,
    ),
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        load_augmented=None,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=None,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="ObjectPaste",
        stop_epoch=15,
        db_sampler=dict(
            dataset_root="data/nuscenes/",
            info_path="data/nuscenes/nuscenes_dbinfos_train.pkl",
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
            classes=[
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
            ],
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
                load_dim=5,
                use_dim=5,
                reduce_beams=32,
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
        resize_lim=[0.9, 1.1],
        rot_lim=[-0.78539816, 0.78539816],
        trans_lim=0.5,
        is_train=True,
    ),
    dict(type="RandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type="ObjectRangeFilter", point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type="ObjectNameFilter",
        classes=[
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
        ],
    ),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="GridMask",
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=augment2d["gridmask"]["prob"],
        fixed_prob=augment2d["gridmask"]["fixed_prob"],
    ),
    dict(type="PointShuffle"),
    dict(
        type="DefaultFormatBundle3D",
        classes=[
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
        ],
    ),
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
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        load_augmented=None,
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        reduce_beams=32,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=None,
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
    dict(type="PointsRangeFilter", point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type="ImageNormalize", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="DefaultFormatBundle3D",
        classes=[
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
        ],
    ),
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
    # C+L 融合（6 相机 + LiDAR + DepthLSS）显存占用大；12GB 单卡建议 1
    # 多卡 24GB 可通过 --cfg-options data.samples_per_gpu=2 增大
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline,
            modality=input_modality,
        ),
    ),
    val=dict(
        pipeline=test_pipeline,
        modality=input_modality,
    ),
    test=dict(
        pipeline=test_pipeline,
        modality=input_modality,
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

lr_config = dict(
    _delete_=True,
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.33333333,
    min_lr_ratio=1.0e-3,
)
momentum_config = None

max_epochs = 6
runner = dict(type="CustomEpochBasedRunner", max_epochs=max_epochs)
evaluation = dict(interval=1, pipeline=test_pipeline)

# 微调时可命令行指定: --load-from work_dirs/bevfusion_lidar_v0075/latest.pth
load_from = None
