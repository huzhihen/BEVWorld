"""nuScenes 检测公共变量（AnchorHead / CenterHead 等配置复用）。"""

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

centerhead_tasks = [
    ["car"],
    ["truck", "construction_vehicle"],
    ["bus", "trailer"],
    ["barrier"],
    ["motorcycle", "bicycle"],
    ["pedestrian", "traffic_cone"],
]
