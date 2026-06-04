"""
BEVFusion + MapTR 联合数据准备入口。

功能：
  调用 nuscenes_joint_converter，将 nuScenes 原始数据转换为联合 pkl，
  单条样本同时包含 BEVFusion 3D 检测标注与 MapTRv2 矢量地图标注。

用法（在 BEVWorld 仓库根目录执行）::

    # mini 集快速验证
    python tools_bevfuison_maptr/create_data.py \\
        --root-path data/nuscenes \\
        --canbus data \\
        --version v1.0-mini \\
        --out-dir data/nuscenes \\
        --extra-tag nuscenes_bevfusion_maptr \\
        --point-cloud-range -15 -10 -10 15 30 10

    # 完整 trainval
    python tools_bevfuison_maptr/create_data.py \\
        --root-path data/nuscenes \\
        --canbus data \\
        --version v1.0-trainval \\
        --out-dir data/nuscenes \\
        --extra-tag nuscenes_bevfusion_maptr

输出文件（在 --out-dir 下）::
    {extra-tag}_map_infos_temporal_train.pkl
    {extra-tag}_map_infos_temporal_val.pkl
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from nuscenes_joint_converter import create_nuscenes_infos  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create joint BEVFusion+MapTR nuScenes pkl for BEVWorld')
    parser.add_argument('--root-path', type=str, default='./data/nuscenes')
    parser.add_argument('--canbus', type=str, default='./data')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--max-sweeps', type=int, default=10)
    parser.add_argument('--maps', nargs='*', default=None)
    parser.add_argument(
        '--point-cloud-range', nargs=6, type=float,
        default=[-15.0, -10.0, -10.0, 15.0, 30.0, 10.0])
    parser.add_argument('--use-ego-origin', action='store_true')
    parser.add_argument('--map-yaw-offset-deg', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='./data/nuscenes')
    parser.add_argument(
        '--extra-tag', type=str, default='nuscenes_bevfusion_maptr')
    parser.add_argument('--max-samples', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.version in ('v1.0-mini', 'v1.0-trainval', 'v1.0-test'):
        create_nuscenes_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=args.version,
            max_sweeps=args.max_sweeps,
            maps=args.maps,
            point_cloud_range=args.point_cloud_range,
            use_ego_origin=args.use_ego_origin,
            map_yaw_offset_deg=args.map_yaw_offset_deg,
            max_samples=args.max_samples,
        )
    else:
        train_version = f'{args.version}-trainval'
        create_nuscenes_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps,
            maps=args.maps,
            point_cloud_range=args.point_cloud_range,
            use_ego_origin=args.use_ego_origin,
            map_yaw_offset_deg=args.map_yaw_offset_deg,
            max_samples=args.max_samples,
        )
        test_version = f'{args.version}-test'
        create_nuscenes_infos(
            root_path=args.root_path,
            out_path=args.out_dir,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps,
            maps=args.maps,
            point_cloud_range=args.point_cloud_range,
            use_ego_origin=args.use_ego_origin,
            map_yaw_offset_deg=args.map_yaw_offset_deg,
            max_samples=args.max_samples,
        )


if __name__ == '__main__':
    main()
