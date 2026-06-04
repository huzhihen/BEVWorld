_base_ = [
    "./bevfusion_maptr_nusc_r50_24ep_w_centerline_fusion.py",
]

data_root = "data/nuscenes/"
info_root = "data/"
train_ann_file = info_root + "nuscenes_bevfusion_maptr_map_infos_temporal_train.pkl"
val_ann_file = info_root + "nuscenes_bevfusion_maptr_map_infos_temporal_val.pkl"
map_ann_file = "data/nuscenes_map_anns_val.json"

load_dim = 5
use_dim = [0, 1, 2, 3, 4]

point_cloud_range = [-15.0, -10.0, -10.0, 15.0, 30.0, 10.0]
object_point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
object_lidar_point_cloud_range = object_point_cloud_range
lidar_point_cloud_range = object_lidar_point_cloud_range

map_voxel_size = [0.15, 0.15, 20.0]
lidar_voxel_size = [0.075, 0.075, 0.2]
dbound = [1.0, 60.0, 0.5]
object_sparse_shape = [1440, 1440, 41]
object_head_grid_size = [1440, 1440, 41]
camera_object_bev_step = 0.6
image_resize_scale = 0.6

grid_config = {
    "x": [point_cloud_range[0], point_cloud_range[3], map_voxel_size[0]],
    "y": [point_cloud_range[1], point_cloud_range[4], map_voxel_size[1]],
    "z": [point_cloud_range[2], point_cloud_range[5], map_voxel_size[2]],
    "depth": dbound,
}

bev_h_ = int((point_cloud_range[4] - point_cloud_range[1]) / map_voxel_size[1] / 2)
bev_w_ = int((point_cloud_range[3] - point_cloud_range[0]) / map_voxel_size[0] / 2)

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
num_object_classes = len(object_classes)

nuscenes_name_mapping = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
}
augment3d = dict(
    scale=[0.95, 1.05],
    rotate=[-0.39269908, 0.39269908],
    translate=0.5,
)

official_nuscenes_anchor_sizes = [
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
]
anchor_assigner_per_class = [
    dict(
        type="MaxIoUAssigner",
        iou_calculator=dict(type="BboxOverlapsNearest3D"),
        pos_iou_thr=0.5,
        neg_iou_thr=0.2,
        min_pos_iou=0.2,
        ignore_iof_thr=-1,
    )
    for _ in object_classes
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

model = dict(
    encoders=dict(
        camera=dict(
            vtransform=dict(
                image_size=((540 + 31) // 32 * 32, (960 + 31) // 32 * 32),
                feature_size=(((540 + 31) // 32 * 32) // 8, ((960 + 31) // 32 * 32) // 8),
                xbound=[object_point_cloud_range[0], object_point_cloud_range[3], camera_object_bev_step],
                ybound=[object_point_cloud_range[1], object_point_cloud_range[4], camera_object_bev_step],
                zbound=[object_point_cloud_range[2], object_point_cloud_range[5], object_point_cloud_range[5] - object_point_cloud_range[2]],
                dbound=dbound,
            ),
        ),
        lidar=dict(
            voxelize=dict(
                point_cloud_range=object_lidar_point_cloud_range,
                voxel_size=lidar_voxel_size,
                max_voxels=[120000, 160000],
            ),
            backbone=dict(
                in_channels=5,
                sparse_shape=object_sparse_shape,
            ),
        ),
    ),
    heads=dict(
        object=dict(
            num_classes=num_object_classes,
            assigner_per_size=True,
            assign_per_class=True,
            anchor_generator=dict(
                ranges=[object_point_cloud_range],
                sizes=official_nuscenes_anchor_sizes,
                reshape_out=False,
            ),
            train_cfg=dict(
                assigner=anchor_assigner_per_class,
                allowed_border=0,
                code_weight=[1.0] * 7,
                pos_weight=-1,
                debug=False,
            ),
        ),
        vectormap=dict(
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_classes=4,
            transformer=dict(
                num_cams=6,
                encoder=dict(
                    pc_range=point_cloud_range,
                    voxel_size=map_voxel_size,
                    dbound=dbound,
                    grid_config=grid_config,
                ),
            ),
            bbox_coder=dict(
                post_center_range=[
                    point_cloud_range[0],
                    point_cloud_range[1],
                    point_cloud_range[0],
                    point_cloud_range[1],
                    point_cloud_range[3],
                    point_cloud_range[4],
                    point_cloud_range[3],
                    point_cloud_range[4],
                ],
                pc_range=point_cloud_range,
                voxel_size=map_voxel_size,
            ),
            positional_encoding=dict(
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
        ),
    ),
)

train_cfg = dict(
    pts=dict(
        grid_size=object_head_grid_size,
        voxel_size=lidar_voxel_size,
        point_cloud_range=object_point_cloud_range,
        out_size_factor=8,
        assigner=dict(pc_range=point_cloud_range),
    )
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="RandomScaleImageMultiViewImage", scales=[image_resize_scale]),
    dict(
        type="NormalizeMultiviewImage",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(type="CustomPointToMultiViewDepth", downsample=1, grid_config=grid_config),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type="GlobalRotScaleTrans",
        resize_lim=augment3d["scale"],
        rot_lim=augment3d["rotate"],
        trans_lim=augment3d["translate"],
        is_train=True,
    ),
    dict(type="RandomFlip3D"),
    dict(type="PointsRangeFilter", point_cloud_range=object_lidar_point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=object_lidar_point_cloud_range),
    dict(type="ObjectNameFilter", classes=object_classes),
    dict(type="PointShuffle"),
    dict(type="PadMultiViewImageDepth", size_divisor=32),
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(type="CustomCollect3D", keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d", "gt_depth"]),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(type="RandomScaleImageMultiViewImage", scales=[image_resize_scale]),
    dict(
        type="NormalizeMultiviewImage",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", with_gt=False, with_label=False, classes=object_classes),
    dict(type="CustomCollect3D", keys=["img", "points"]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        aux_seg=dict(use_aux_seg=True, bev_seg=True, pv_seg=True, seg_classes=1, feat_down_sample=16, pv_thickness=1),
        test_mode=False,
        with_velocity=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        map_classes=["divider", "ped_crossing", "boundary", "centerline"],
        queue_length=1,
        box_type_3d="LiDAR",
    ),
    val=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        with_velocity=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        map_classes=["divider", "ped_crossing", "boundary", "centerline"],
        queue_length=1,
        samples_per_gpu=1,
    ),
    test=dict(
        data_root=data_root,
        ann_file=val_ann_file,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        with_velocity=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        map_classes=["divider", "ped_crossing", "boundary", "centerline"],
        queue_length=1,
    ),
)

evaluation = dict(interval=4, pipeline=test_pipeline, metric=["bbox", "chamfer"])
