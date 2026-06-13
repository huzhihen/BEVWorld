#!/usr/bin/env bash
#
# BEVFusion 多卡测试 / 评估启动脚本
#
# 用法:
#   bash tools/dist_test.sh CONFIG CHECKPOINT GPUS [extra test.py args...]
#
# 示例:
#   bash tools/dist_test.sh projects/configs/bevfusion/bevfusion_nusc_lidar_transfusion_voxel0075.py \
#     work_dirs/bevfusion_lidar_v0075/latest.pth 8 --eval bbox

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
