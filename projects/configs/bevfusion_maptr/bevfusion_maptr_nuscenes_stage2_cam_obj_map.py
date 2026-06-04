_base_ = [
    "./bevfusion_maptr_nusc_r50_24ep_w_centerline_fusion.py",
]

load_from = None

model = dict(
    freeze_modules=["encoders.lidar"],
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "encoders.camera.backbone": dict(lr_mult=0.1),
            "encoders.lidar": dict(lr_mult=0.0),
        },
    ),
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
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
checkpoint_config = dict(max_keep_ckpts=3, interval=2)
find_unused_parameters = True

log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
evaluation = dict(interval=4, metric=["bbox", "chamfer"])
