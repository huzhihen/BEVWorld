plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = None
load_from = None
resume_from = None
workflow = [("train", 1)]

data_root = "data/nuscenes/"
info_root = "data/nuscenes/"
train_ann_file = info_root + "nuscenes_bevfusion_maptr_map_infos_temporal_train.pkl"
val_ann_file = info_root + "nuscenes_bevfusion_maptr_map_infos_temporal_val.pkl"
ann_file = val_ann_file
map_ann_file = "data/nuscenes_map_anns_val.json"

image_size = (900, 1600)
image_resize_scale = 0.5
image_size_padded = (
    int(image_size[0] * image_resize_scale + 31) // 32 * 32,
    int(image_size[1] * image_resize_scale + 31) // 32 * 32,
)
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

grid_config = {
    "x": [point_cloud_range[0], point_cloud_range[3], map_voxel_size[0]],
    "y": [point_cloud_range[1], point_cloud_range[4], map_voxel_size[1]],
    "z": [point_cloud_range[2], point_cloud_range[5], map_voxel_size[2]],
    "depth": dbound,
}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

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
map_classes = [
    "divider",
    "ped_crossing",
    "boundary",
    "centerline",
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
augment3d = dict(
    scale=[0.95, 1.05],
    rotate=[-0.39269908, 0.39269908],
    translate=0.5,
)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
bev_h_ = int((point_cloud_range[4] - point_cloud_range[1]) / map_voxel_size[1] / 2)
bev_w_ = int((point_cloud_range[3] - point_cloud_range[0]) / map_voxel_size[0] / 2)
num_vec = 70
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)
num_object_classes = len(object_classes)
queue_length = 1

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    seg_classes=1,
    feat_down_sample=16,
    pv_thickness=1,
)

model = dict(
    type="BEVFusionMapTR",
    use_lidar_for_vectormap=False,
    encoders=dict(
        camera=dict(
            backbone=dict(
                type="ResNet",
                depth=50,
                out_indices=(1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=False),
                norm_eval=True,
                style="pytorch",
                # 相对路径：在 bevfusion 仓库根目录下执行训练时解析为 bevfusion/ckpts/...
                init_cfg=dict(
                    type="Pretrained",
                    checkpoint="ckpts/resnet50-19c8e357.pth",
                ),
            ),
            neck=dict(
                type="GeneralizedLSSFPN",
                in_channels=[512, 1024, 2048],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type="BN2d", requires_grad=True),
                act_cfg=dict(type="ReLU", inplace=True),
                upsample_cfg=dict(mode="bilinear", align_corners=False),
            ),
            vtransform=dict(
                type="DepthLSSTransform",
                in_channels=_dim_,
                out_channels=80,
                image_size=image_size_padded,
                feature_size=(image_size_padded[0] // 8, image_size_padded[1] // 8),
                xbound=[object_point_cloud_range[0], object_point_cloud_range[3], camera_object_bev_step],
                ybound=[object_point_cloud_range[1], object_point_cloud_range[4], camera_object_bev_step],
                zbound=[object_point_cloud_range[2], object_point_cloud_range[5], object_point_cloud_range[5] - object_point_cloud_range[2]],
                dbound=dbound,
                downsample=2,
            ),
        ),
        lidar=dict(
            voxelize=dict(
                max_num_points=10,
                point_cloud_range=object_lidar_point_cloud_range,
                voxel_size=lidar_voxel_size,
                max_voxels=[120000, 160000],
            ),
            backbone=dict(
                type="SparseEncoder",
                in_channels=5,
                sparse_shape=object_sparse_shape,
                output_channels=128,
                order=("conv", "norm", "act"),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
                block_type="basicblock",
            ),
        ),
    ),
    fuser=dict(
        type="ConvFuser",
        in_channels=[80, 256],
        out_channels=256,
    ),
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
        object=dict(
            type="Anchor3DHead",
            in_channels=512,
            feat_channels=512,
            num_classes=num_object_classes,
            use_direction_classifier=True,
            assigner_per_size=True,
            assign_per_class=True,
            anchor_generator=dict(
                type="AlignedAnchor3DRangeGenerator",
                ranges=[object_point_cloud_range],
                sizes=official_nuscenes_anchor_sizes,
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
            loss_dir=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=0.2,
            ),
            train_cfg=dict(
                assigner=anchor_assigner_per_class,
                allowed_border=0,
                code_weight=[1.0] * 7,
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
        vectormap=dict(
            type="MapTRv2Head",
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_vec_one2one=num_vec,
            num_vec_one2many=300,
            k_one2many=6,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
            dir_interval=1,
            query_embed_type="instance_pts",
            transform_method="minmax",
            gt_shift_pts_pattern="v2",
            num_classes=num_map_classes,
            in_channels=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            code_size=2,
            code_weights=[1.0, 1.0, 1.0, 1.0],
            aux_seg=aux_seg_cfg,
            transformer=dict(
                type="MapTRPerceptionTransformer",
                num_cams=6,
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                modality="vision",
                feat_down_sample_indice=-1,
                encoder=dict(
                    type="LSSTransform",
                    in_channels=_dim_,
                    out_channels=_dim_,
                    feat_down_sample=16,
                    pc_range=point_cloud_range,
                    voxel_size=map_voxel_size,
                    dbound=dbound,
                    downsample=2,
                    loss_depth_weight=3.0,
                    depthnet_cfg=dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
                    grid_config=grid_config,
                ),
                decoder=dict(
                    type="MapTRDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DecoupledDetrTransformerDecoderLayer",
                        num_vec=num_vec,
                        num_pts_per_vec=fixed_ptsnum_per_pred_line,
                        attn_cfgs=[
                            dict(type="MultiheadAttention", embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type="MultiheadAttention", embed_dims=_dim_, num_heads=8, dropout=0.1),
                            dict(type="CustomMSDeformableAttention", embed_dims=_dim_, num_levels=1),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
            ),
            bbox_coder=dict(
                type="MapTRNMSFreeCoder",
                post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
                pc_range=point_cloud_range,
                max_num=50,
                voxel_size=map_voxel_size,
                num_classes=num_map_classes,
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=0.0),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
            loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
            loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
            loss_seg=dict(type="SimpleLoss", pos_weight=4.0, loss_weight=1.0),
            loss_pv_seg=dict(type="SimpleLoss", pos_weight=1.0, loss_weight=2.0),
        ),
    ),
    loss_scale=dict(object=1.0, vectormap=1.0),
)

train_cfg = dict(
    pts=dict(
        grid_size=object_head_grid_size,
        voxel_size=lidar_voxel_size,
        point_cloud_range=object_point_cloud_range,
        out_size_factor=8,
        assigner=dict(
            type="MapTRAssigner",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBoxL1Cost", weight=0.0, box_format="xywh"),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=0.0),
            pts_cost=dict(type="OrderedPtsL1Cost", weight=5),
            pc_range=point_cloud_range,
        ),
    )
)

dataset_type = "BEVFusionMapTRJointDataset"

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="RandomScaleImageMultiViewImage", scales=[image_resize_scale]),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=load_dim,
        use_dim=use_dim,
    ),
    dict(type="CustomPointToMultiViewDepth", downsample=1, grid_config=grid_config),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
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
    dict(
        type="CustomCollect3D",
        keys=["img", "points", "gt_bboxes_3d", "gt_labels_3d", "gt_depth"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=load_dim,
        use_dim=use_dim,
    ),
    dict(type="RandomScaleImageMultiViewImage", scales=[image_resize_scale]),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="DefaultFormatBundle3D",
        with_gt=False,
        with_label=False,
        classes=object_classes,
    ),
    dict(type="CustomCollect3D", keys=["img", "points"]),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        aux_seg=aux_seg_cfg,
        test_mode=False,
        with_velocity=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        queue_length=queue_length,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        with_velocity=False,
        queue_length=queue_length,
        samples_per_gpu=1,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        map_ann_file=map_ann_file,
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        map_classes=map_classes,
        classes=object_classes,
        name_mapping=nuscenes_name_mapping,
        eval_version="detection_cvpr_2019",
        modality=input_modality,
        with_velocity=False,
        queue_length=queue_length,
    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={"encoders.camera.backbone": dict(lr_mult=0.1)}),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

total_epochs = 24
evaluation = dict(
    interval=4,
    pipeline=test_pipeline,
    metric=["bbox", "chamfer"],
    save_best="NuscMap_chamfer/mAP",
    rule="greater",
)

runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
# Static loss_scale=512 easily overflows when the dense anchor object loss is
# large at the beginning of training. Dynamic scaling backs off on overflow.
fp16 = dict(loss_scale=dict(init_scale=64.0, growth_interval=2000))
checkpoint_config = dict(max_keep_ckpts=1, interval=2)
find_unused_parameters = True
