_base_ = [
    "./bevfusion_maptr_nusc_r50_24ep_w_centerline_fusion.py",
]

cudnn_benchmark = True
data_root = "data/nuscenes/"
info_root = "data/"
train_ann_file = info_root + "nuscenes_maptr_map_infos_temporal_train.pkl"
val_ann_file = info_root + "nuscenes_maptr_map_infos_temporal_val.pkl"
load_dim = 5
use_dim = [0, 1, 2, 3, 4]
sweeps_num = 4
object_lidar_point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
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

model = dict(
    encoders=dict(camera=None),
    fuser=None,
    decoder=dict(backbone=dict(in_channels=256)),
    heads=dict(vectormap=None),
    loss_scale=dict(object=1.0),
)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=load_dim,
        use_dim=use_dim,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
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
    dict(type="DefaultFormatBundle3D", classes=object_classes),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]

test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=load_dim, use_dim=use_dim),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=sweeps_num,
        load_dim=load_dim,
        use_dim=use_dim,
        pad_empty_sweeps=True,
        remove_close=True,
    ),
    dict(type="DefaultFormatBundle3D", with_gt=False, with_label=False, classes=object_classes),
    dict(type="Collect3D", keys=["points"]),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        _delete_=True,
        type="CBGSDataset",
        dataset=dict(
            type="NuScenesDataset",
            dataset_root=data_root,
            ann_file=train_ann_file,
            pipeline=train_pipeline,
            object_classes=object_classes,
            name_mapping=nuscenes_name_mapping,
            eval_version="detection_cvpr_2019",
            modality=input_modality,
            test_mode=False,
            with_velocity=False,
            use_valid_flag=True,
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        _delete_=True,
        type="NuScenesDataset",
        dataset_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        object_classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        test_mode=True,
        with_velocity=False,
        box_type_3d="LiDAR",
    ),
    test=dict(
        _delete_=True,
        type="NuScenesDataset",
        dataset_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        object_classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        test_mode=True,
        with_velocity=False,
        box_type_3d="LiDAR",
    ),
)

optimizer = dict(type="AdamW", lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 24
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = dict(max_keep_ckpts=3, interval=2)
find_unused_parameters = False
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])

# 动态 loss scale：NaN 时自动减半，避免 fp16 静态 scale 导致每步跳过更新
# _delete_=True 是因为 base config 里 loss_scale 是 float，类型不同必须整体替换
fp16 = dict(_delete_=True, loss_scale=dict(init_scale=64.0, growth_interval=2000))
evaluation = dict(
    interval=4,
    pipeline=test_pipeline,
    metric="bbox",
    save_best="object/map",
    rule="greater",
)
