"""
BEVFusion nuScenes LiDAR-only Anchor3DHead 配置（SECFPN, voxel 0.075）。

训练::

    bash tools/dist_train.sh \\
      projects/configs/bevfusion/bevfusion_nusc_lidar_anchor_voxel0075.py 8 \\
      --work-dir work_dirs/bevfusion_lidar_anchor_v0075

评估::

    bash tools/dist_test.sh \\
      projects/configs/bevfusion/bevfusion_nusc_lidar_anchor_voxel0075.py \\
      work_dirs/bevfusion_lidar_anchor_v0075/latest.pth 8 --eval bbox
"""

_base_ = ["./bevfusion_nusc_lidar_transfusion_voxel0075.py"]

# Anchor3DHead 使用 7 维框，需关闭 nuScenes 默认的速度维 (with_velocity=True → 9 维)
data = dict(
    train=dict(dataset=dict(with_velocity=False)),
    val=dict(with_velocity=False),
    test=dict(with_velocity=False),
)

_anchor_assigner = dict(
    type="MaxIoUAssigner",
    iou_calculator=dict(type="BboxOverlapsNearest3D"),
    pos_iou_thr=0.5,
    neg_iou_thr=0.2,
    min_pos_iou=0.2,
    ignore_iof_thr=-1,
)

model = dict(
    heads=dict(
        object=dict(
            _delete_=True,
            type="Anchor3DHead",
            in_channels=512,
            feat_channels=512,
            num_classes=10,
            use_direction_classifier=True,
            assigner_per_size=True,
            assign_per_class=True,
            anchor_generator=dict(
                type="AlignedAnchor3DRangeGenerator",
                ranges=[[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]],
                sizes=[
                    [1.96, 4.64, 1.74],
                    [2.52, 6.96, 2.86],
                    [2.82, 6.55, 3.20],
                    [2.95, 11.21, 3.49],
                    [2.93, 12.28, 3.88],
                    [2.51, 0.50, 0.99],
                    [0.77, 2.10, 1.46],
                    [0.61, 1.70, 1.30],
                    [0.67, 0.73, 1.77],
                    [0.41, 0.41, 1.07],
                ],
                rotations=[0, 1.57],
                reshape_out=False,
            ),
            diff_rad_by_sin=True,
            dir_offset=0.7854,
            dir_limit_offset=0,
            bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder", code_size=7),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="SmoothL1Loss", beta=0.111111, loss_weight=1.0),
            loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
            train_cfg=dict(
                assigner=[_anchor_assigner] * 10,
                allowed_border=0,
                code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                pos_weight=-1,
                debug=False,
            ),
            test_cfg=dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=2000,
                nms_thr=0.2,
                score_thr=0.3,
                min_bbox_size=0,
                max_num=500,
            ),
        ),
    ),
)
