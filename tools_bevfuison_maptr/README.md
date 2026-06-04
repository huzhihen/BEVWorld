# tools_bevfuison_maptr

BEVFusion + MapTRv2 联合工具链（数据 pkl、训练、测试）。

## 1. 生成联合 pkl

```bash
cd /path/to/BEVWorld

python tools_bevfuison_maptr/create_data.py \
  --root-path data/nuscenes \
  --canbus data \
  --version v1.0-trainval \
  --out-dir data/nuscenes \
  --extra-tag nuscenes_bevfusion_maptr \
  --point-cloud-range -15 -10 -10 15 30 10
```

输出：`data/nuscenes/nuscenes_bevfusion_maptr_map_infos_temporal_{train,val}.pkl`

## 2. 训练

```bash
torchpack dist-run -np 4 python tools_bevfuison_maptr/maptr_train.py \
  projects/configs/bevfusion_maptr/bevfusion_maptr_nusc_r50_24ep_w_centerline_fusion.py \
  --run-dir work_dirs/bevfusion_maptr_nusc
```

## 3. 测试

```bash
python tools_bevfuison_maptr/maptr_test.py \
  projects/configs/bevfusion_maptr/bevfusion_maptr_nusc_r50_24ep_w_centerline_fusion.py \
  work_dirs/bevfusion_maptr_nusc/latest.pth \
  --eval bbox chamfer
```

配置与模型代码见 `projects/configs/bevfusion_maptr/` 与 `projects/mmdet3d_plugin/bevfusion_maptr/`。
