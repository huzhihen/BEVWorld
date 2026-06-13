checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)

load_from = None
resume_from = None
cudnn_benchmark = False
fp16 = dict(loss_scale=dict(growth_interval=2000))

dist_params = dict(backend="nccl")
log_level = "INFO"
workflow = [("train", 1)]
