#!/usr/bin/env bash
#
# BEVFusion 多卡训练启动脚本（OpenMMLab 风格，与 tools_maptr/dist_train.sh 一致）
#
# 用法:
#   bash tools/dist_train.sh CONFIG GPUS [extra train.py args...]
#
# 示例:
#   bash tools/dist_train.sh projects/configs/bevfusion/bevfusion_nusc_lidar_transfusion_voxel0075.py 8 \
#     --work-dir work_dirs/bevfusion_lidar_v0075
#
# 仍兼容旧版 YAML:
#   bash tools/dist_train.sh configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml 8

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
