"""
BEVFusion nuScenes Camera + LiDAR 融合 CenterHead 配置（Swin-T, voxel 0.075）。

训练::

    bash tools/dist_train.sh \\
      projects/configs/bevfusion/bevfusion_nusc_camera_lidar_centerhead_swint_voxel0075.py 8 \\
      --work-dir work_dirs/bevfusion_camera_lidar_centerhead_v0075 \\
      --cfg-options model.encoders.camera.backbone.init_cfg.checkpoint=pretrained/swint-nuimages-pretrained.pth
"""

_base_ = ["./bevfusion_nusc_camera_lidar_transfusion_swint_voxel0075.py"]

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
grid_size = [1440, 1440, 1]

model = dict(
    heads=dict(
        object=dict(
            _delete_=True,
            type="CenterHead",
            in_channels=512,
            tasks=[
                ["car"],
                ["truck", "construction_vehicle"],
                ["bus", "trailer"],
                ["barrier"],
                ["motorcycle", "bicycle"],
                ["pedestrian", "traffic_cone"],
            ],
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
