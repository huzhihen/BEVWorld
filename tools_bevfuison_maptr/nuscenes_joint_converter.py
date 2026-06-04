"""Joint BEVFusion + MapTRv2 data converter for BEVWorld (nuScenes / Westwell-like).

参考 bevfusion/tools/3dod_maptr/westwell_joint_converter.py，在 BEVWorld 中生成
BEVFusion + MapTR 联合训练所需的 pkl。

Generates a single pkl containing:
  - Sensor metadata aligned with BEVFusion conventions (camera_intrinsics, location)
  - 3D object detection GT (gt_boxes, gt_names, gt_velocity, ...)  -- BEVFusion
  - Vectorized HD-map GT
    (annotation: divider/ped_crossing/boundary/centerline/
     bar_markings/bar_markings_curve) -- MapTR
  - Temporal info (can_bus, prev, next, scene_token, frame_idx) -- MapTR

Field naming convention:
  BEVFusion keys are primary; MapTR-only keys keep MapTR names.
  For fields shared by both, dual keys are written for backward compatibility:
    camera_intrinsics (BEVFusion) + cam_intrinsic (MapTR compat)
    location (BEVFusion) + map_location (MapTR compat)
"""

import argparse
import os
import sys
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Tuple

import mmcv
import networkx as nx
import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from shapely import affinity, ops
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    box,
)


# ---------------------------------------------------------------------------
# Map helpers (from MapTR)
# ---------------------------------------------------------------------------

class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __ini__(self, *args, **kwargs):
        super(self, CNuScenesMapExplorer).__init__(*args, **kwargs)

    def _get_centerline(self, patch_box, patch_angle, layer_name,
                        return_token=False,
                        patch_center_local=(0.0, 0.0)):
        if layer_name not in ['lane', 'lane_connector']:
            raise ValueError('{} is not a centerline layer'.format(layer_name))
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.get_patch_coord(patch_box, patch_angle)
        records = getattr(self.map_api, layer_name)
        centerline_dict = dict()
        for record in records:
            if record['polygon_token'] is None:
                continue
            polygon = self.map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    centerline = self.map_api.discretize_lanes(record, 0.5)
                    centerline = list(
                        self.map_api.discretize_lanes(
                            [record['token']], 0.5
                        ).values()
                    )[0]
                    centerline = LineString(
                        np.array(centerline)[:, :2].round(3)
                    )
                    if centerline.is_empty:
                        continue
                    centerline = centerline.intersection(patch)
                    if not centerline.is_empty:
                        centerline = to_patch_coord(
                            centerline, patch_angle, patch_x, patch_y,
                            patch_center_local,
                        )
                        record_dict = dict(
                            centerline=centerline,
                            token=record['token'],
                            incoming_tokens=self.map_api.get_incoming_lane_ids(
                                record['token']
                            ),
                            outgoing_tokens=self.map_api.get_outgoing_lane_ids(
                                record['token']
                            ),
                        )
                        centerline_dict.update(
                            {record['token']: record_dict}
                        )
        return centerline_dict


def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y,
                   patch_center_local=(0.0, 0.0)):
    new_polygon = affinity.rotate(
        new_polygon, -patch_angle, origin=(patch_x, patch_y),
        use_radians=False,
    )
    local_x, local_y = patch_center_local
    new_polygon = affinity.affine_transform(
        new_polygon,
        [1.0, 0.0, 0.0, 1.0, -patch_x + local_x, -patch_y + local_y],
    )
    return new_polygon


def get_patch_size_and_center(point_cloud_range):
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    patch_h = y_max - y_min
    patch_w = x_max - x_min
    patch_center_local = ((x_min + x_max) * 0.5,
                          (y_min + y_max) * 0.5)
    return (patch_h, patch_w), patch_center_local


# ---------------------------------------------------------------------------
# NuScenes scene/sensor utilities
# ---------------------------------------------------------------------------

def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        scene_not_exist = False
        while True:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
            break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    if nusc_can_bus is None:
        return np.zeros(18)
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except Exception:
        return np.zeros(18)
    can_bus = []
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])
    can_bus.extend([0., 0.])
    return np.array(can_bus)


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t,
                      e2g_r_mat, sensor_type='lidar', inv_flag=False):
    """BEVFusion version with inv_flag support for UK/HIT sites."""
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get(
        'calibrated_sensor', sd_rec['calibrated_sensor_token']
    )
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    if inv_flag and sensor_type != 'lidar':
        R_cur = R.T
        T_cur = T
        R_inv = R_cur.T
        T_inv = -T_cur @ R_cur
        sweep['sensor2lidar_rotation'] = R_inv
        sweep['sensor2lidar_translation'] = T_inv
    else:
        sweep['sensor2lidar_rotation'] = R.T
        sweep['sensor2lidar_translation'] = T
    return sweep


# ---------------------------------------------------------------------------
# VectorizedLocalMap
# ---------------------------------------------------------------------------

class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'bar_markings': 4,
        'bar_markings_curve': 5,
        'others': -1,
    }

    def __init__(self, nusc_map, map_explorer, patch_size,
                 patch_center_local=(0.0, 0.0),
                 map_classes=('divider', 'ped_crossing', 'boundary',
                              'centerline', 'bar_markings',
                              'bar_markings_curve'),
                 line_classes=('road_divider', 'lane_divider'),
                 ped_crossing_classes=('ped_crossing',),
                 contour_classes=('road_segment', 'lane'),
                 centerline_classes=('lane_connector', 'lane'),
                 polygon_marking_classes=('bar_markings',
                                          'bar_markings_curve'),
                 use_simplify=True,
                 map_yaw_offset_deg=0.0):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.centerline_classes = centerline_classes
        self.polygon_marking_classes = polygon_marking_classes
        self.patch_size = patch_size
        self.patch_center_local = tuple(float(v) for v in patch_center_local)
        self.local_x_min = self.patch_center_local[0] - self.patch_size[1] * 0.5
        self.local_x_max = self.patch_center_local[0] + self.patch_size[1] * 0.5
        self.local_y_min = self.patch_center_local[1] - self.patch_size[0] * 0.5
        self.local_y_max = self.patch_center_local[1] + self.patch_size[0] * 0.5
        self.map_yaw_offset_deg = float(map_yaw_offset_deg)

    def _global_patch_center(self, map_pose, patch_angle):
        theta = np.deg2rad(patch_angle)
        local_x, local_y = self.patch_center_local
        global_x = (map_pose[0]
                    + np.cos(theta) * local_x
                    - np.sin(theta) * local_y)
        global_y = (map_pose[1]
                    + np.sin(theta) * local_x
                    + np.cos(theta) * local_y)
        return global_x, global_y

    def _to_local_coord(self, geom, patch_angle, patch_x, patch_y):
        return to_patch_coord(
            geom, patch_angle, patch_x, patch_y, self.patch_center_local)

    def gen_vectorized_samples(self, lidar2global_translation,
                               lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_angle = (quaternion_yaw(rotation) / np.pi * 180
                       + self.map_yaw_offset_deg)
        patch_x, patch_y = self._global_patch_center(map_pose, patch_angle)
        patch_box = (patch_x, patch_y,
                 self.patch_size[0], self.patch_size[1])
        map_dict = {vec_class: [] for vec_class in self.vec_classes}
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(
                            np.array(instance.coords))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes)
                ped_instance_list = self.polygon_geoms_to_instances(
                    ped_geom, 'ped_crossing')
                for instance in ped_instance_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'centerline':
                arcline_path_3 = getattr(
                    self.map_explorer.map_api, 'arcline_path_3', None)
                if isinstance(arcline_path_3, dict):
                    centerline_geom = self.get_centerline_geom(
                        patch_box, patch_angle, self.centerline_classes)
                    centerline_list = self.centerline_geoms_to_instances(
                        centerline_geom)
                    for instance in centerline_list:
                        map_dict[vec_class].append(
                            np.array(instance.coords))
            elif vec_class in self.polygon_marking_classes:
                marking_geom = self.get_map_geom(
                    patch_box, patch_angle, (vec_class,))
                marking_instance_list = self.polygon_geoms_to_instances(
                    marking_geom, vec_class)
                for instance in marking_instance_list:
                    map_dict[vec_class].append(np.array(instance.coords))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        return map_dict

    def get_centerline_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.centerline_classes:
                layer_centerline_dict = self.map_explorer._get_centerline(
                    patch_box, patch_angle, layer_name, return_token=False,
                    patch_center_local=self.patch_center_local)
                if len(layer_centerline_dict.keys()) == 0:
                    continue
                map_geom.update(layer_centerline_dict)
        return map_geom

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(
                    patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_polygon_layer_line(
                    patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
            elif layer_name in self.polygon_marking_classes:
                geoms = self.get_polygon_layer_line(
                    patch_box, patch_angle, layer_name)
                map_geom[layer_name] = geoms
        return map_geom

    def get_divider_line(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError('{} is not a line layer'.format(layer_name))
        if layer_name == 'traffic_light':
            return None
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = self._to_local_coord(
                    new_line, patch_angle, patch_x, patch_y)
                line_list.append(new_line)
        return line_list

    def get_contour_line(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError(
                '{} is not a polygonal layer'.format(layer_name))
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        records = getattr(self.map_explorer.map_api, layer_name)
        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [
                    self.map_explorer.map_api.extract_polygon(pt)
                    for pt in record['polygon_tokens']
                ]
                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = self._to_local_coord(
                            new_polygon, patch_angle, patch_x, patch_y)
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        else:
            for record in records:
                polygon = self.map_explorer.map_api.extract_polygon(
                    record['polygon_token'])
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = self._to_local_coord(
                            new_polygon, patch_angle, patch_x, patch_y)
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        return polygon_list

    def get_polygon_layer_line(self, patch_box, patch_angle, layer_name):
        if not hasattr(self.map_explorer.map_api, layer_name):
            return []
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            polygon_tokens = []
            polygon_token = record.get('polygon_token')
            if polygon_token:
                polygon_tokens.append(polygon_token)
            polygon_tokens.extend(record.get('polygon_tokens', []) or [])
            for token in polygon_tokens:
                polygon = self.map_explorer.map_api.extract_polygon(token)
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = self._to_local_coord(
                            new_polygon, patch_angle, patch_x, patch_y)
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        return polygon_list

    def get_ped_crossing_line(self, patch_box, patch_angle):
        return self.get_polygon_layer_line(
            patch_box, patch_angle, 'ped_crossing')

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom.items():
            one_type_instances = self._one_type_line_geom_to_instances(
                a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances
        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def _polygon_union_to_instances(self, polygon_list, local_patch):
        if len(polygon_list) == 0:
            return []
        union_segments = ops.unary_union(polygon_list)
        if union_segments.is_empty:
            return []
        exteriors = []
        interiors = []
        if union_segments.geom_type == 'Polygon':
            polygons = [union_segments]
        elif union_segments.geom_type == 'MultiPolygon':
            polygons = list(union_segments.geoms)
        elif union_segments.geom_type == 'GeometryCollection':
            polygons = []
            for geom in union_segments.geoms:
                if geom.geom_type == 'Polygon':
                    polygons.append(geom)
                elif geom.geom_type == 'MultiPolygon':
                    polygons.extend(list(geom.geoms))
        else:
            return []
        for poly in polygons:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)
        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        return self._one_type_line_geom_to_instances(results)

    def polygon_geoms_to_instances(self, polygon_geom, layer_name):
        layer_geoms = polygon_geom.get(layer_name, [])
        local_patch = box(self.local_x_min - 0.2,
                  self.local_y_min - 0.2,
                  self.local_x_max + 0.2,
                  self.local_y_max + 0.2)
        return self._polygon_union_to_instances(layer_geoms, local_patch)

    def ped_poly_geoms_to_instances(self, ped_geom):
        return self.polygon_geoms_to_instances(ped_geom, 'ped_crossing')

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom.get('road_segment', [])
        lanes = polygon_geom.get('lane', [])
        polygon_list = []
        if len(roads) > 0:
            polygon_list.append(ops.unary_union(roads))
        if len(lanes) > 0:
            polygon_list.append(ops.unary_union(lanes))
        local_patch = box(self.local_x_min + 0.2,
                  self.local_y_min + 0.2,
                  self.local_x_max - 0.2,
                  self.local_y_max - 0.2)
        return self._polygon_union_to_instances(polygon_list, local_patch)

    def centerline_geoms_to_instances(self, geoms_dict):
        centerline_geoms_list, _ = self.union_centerline(geoms_dict)
        return self._one_type_line_geom_to_instances(centerline_geoms_list)

    def union_centerline(self, centerline_geoms):
        pts_G = nx.DiGraph()
        junction_pts_list = []
        for key, value in centerline_geoms.items():
            centerline_geom = value['centerline']
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(
                    centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(
                    centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx in range(len(single_geom_pts) - 1):
                        pts_G.add_edge(
                            tuple(single_geom_pts[idx]),
                            tuple(single_geom_pts[idx + 1]),
                        )
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx in range(len(centerline_pts) - 1):
                    pts_G.add_edge(
                        tuple(centerline_pts[idx]),
                        tuple(centerline_pts[idx + 1]),
                    )
            else:
                raise NotImplementedError
            valid_incoming_num = 0
            for pred in value['incoming_tokens']:
                if pred in centerline_geoms.keys():
                    valid_incoming_num += 1
                    pred_geom = centerline_geoms[pred]['centerline']
                    if pred_geom.geom_type == 'MultiLineString':
                        pred_pt = np.array(
                            pred_geom.geoms[-1].coords).round(3)[-1]
                    else:
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]
                    pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))
            valid_outgoing_num = 0
            for succ in value['outgoing_tokens']:
                if succ in centerline_geoms.keys():
                    valid_outgoing_num += 1
                    succ_geom = centerline_geoms[succ]['centerline']
                    if succ_geom.geom_type == 'MultiLineString':
                        succ_pt = np.array(
                            succ_geom.geoms[0].coords).round(3)[0]
                    else:
                        succ_pt = np.array(succ_geom.coords).round(3)[0]
                    pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))
        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            for leaf in leaves:
                try:
                    paths = nx.all_simple_paths(pts_G, root, leaf)
                    all_paths.extend(paths)
                except (nx.NodeNotFound, nx.NetworkXNoPath):
                    continue
        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_G


def obtain_vectormap(nusc_maps, map_explorer, info, point_cloud_range,
                     use_ego_origin=False, map_yaw_offset_deg=0.0):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(
        info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = info['ego2global_translation']
    lidar2global = ego2global if use_ego_origin else (ego2global @ lidar2ego)
    lidar2global_translation = list(lidar2global[:3, 3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

    location = info['map_location']
    patch_size, patch_center_local = get_patch_size_and_center(
        point_cloud_range)
    info['map_point_cloud_range'] = list(point_cloud_range)
    info['map_patch_size'] = list(patch_size)
    info['map_patch_center_offset'] = list(patch_center_local)
    vector_map = VectorizedLocalMap(
        nusc_maps[location], map_explorer[location], patch_size,
        patch_center_local=patch_center_local,
        map_yaw_offset_deg=map_yaw_offset_deg)
    map_anns = vector_map.gen_vectorized_samples(
        lidar2global_translation, lidar2global_rotation)
    info['annotation'] = map_anns
    return info


# ---------------------------------------------------------------------------
# Core: merged _fill_trainval_infos
# ---------------------------------------------------------------------------

def _fill_trainval_infos(nusc, nusc_can_bus, nusc_maps, map_explorer,
                         train_scenes, val_scenes,
                         test=False, max_sweeps=10,
                         point_cloud_range=None,
                         use_ego_origin=False,
                         map_yaw_offset_deg=0.0,
                         cam_channel_map=None,
                         path_substring_map=None,
                         site_name=None,
                         max_samples=None):
    if point_cloud_range is None:
        point_cloud_range = [-15.0, -10.0, -10.0, 15.0, 30.0, 10.0]

    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    camera_types = None

    _remap_dir = osp.dirname(osp.abspath(__file__))
    if _remap_dir not in sys.path:
        sys.path.insert(0, _remap_dir)
    from westwell_cam_remap_util import (
        CAMS as WESTWELL_CAMS,
        infer_canonical_cam_name,
        remap_nuscenes_channel,
    )

    cam_channel_map = cam_channel_map or {}
    path_substring_map = path_substring_map or {}
    warned_unknown_cam = set()
    warned_collision = set()
    inv_flag = site_name in ('uk', 'hit')

    samples = nusc.sample if max_samples is None else nusc.sample[:max_samples]
    for sample in mmcv.track_iter_progress(samples):
        map_location = nusc.get(
            'log', nusc.get('scene', sample['scene_token'])['log_token']
        )['location']

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get(
            'calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        lidar_path = str(lidar_path)
        if lidar_path.endswith('.pcd'):
            lidar_path = lidar_path.replace('.pcd', '.bin')
        mmcv.check_file_exist(lidar_path)

        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)

        info = {
            # -- BEVFusion core --
            'lidar_path': lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'location': map_location,
            # -- MapTR temporal --
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,
            'scene_token': sample['scene_token'],
            # -- MapTR compat alias --
            'map_location': map_location,
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # ---- Auto-detect camera channels (first sample only) ----
        if camera_types is None:
            camera_types = []
            for channel, token in sample['data'].items():
                try:
                    sd_rec_c = nusc.get('sample_data', token)
                except Exception:
                    continue
                if sd_rec_c.get('sensor_modality', None) == 'camera':
                    camera_types.append(channel)
            print(f'[joint_converter] Detected camera channels: '
                  f'{camera_types}')

        # ---- Camera info with remap to westwell 8-slot ----
        cams_merged = {}
        for cam in camera_types:
            if cam not in sample['data']:
                continue
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic_raw = nusc.get_sample_data(cam_token)

            cam_intrinsic = np.array(cam_intrinsic_raw)
            if cam_intrinsic.ndim == 1 and len(cam_intrinsic) == 9:
                cam_intrinsic = cam_intrinsic.reshape(3, 3)

            cam_info = obtain_sensor2top(
                nusc, cam_token, l2e_t, l2e_r_mat,
                e2g_t, e2g_r_mat, cam, inv_flag)
            cam_info['camera_intrinsics'] = cam_intrinsic
            cam_info['cam_intrinsic'] = cam_intrinsic

            canonical = remap_nuscenes_channel(cam, cam_channel_map)
            if canonical not in WESTWELL_CAMS:
                inferred = infer_canonical_cam_name(
                    str(cam_path), path_substring_map)
                if inferred:
                    canonical = inferred
            if canonical not in WESTWELL_CAMS:
                tag = (cam, canonical)
                if tag not in warned_unknown_cam:
                    print(
                        f'[joint_converter] warning: channel "{cam}" '
                        f'unmapped to canonical 8-slot, keeping '
                        f'"{canonical}"')
                    warned_unknown_cam.add(tag)
            if canonical in cams_merged:
                col_key = (cam, canonical)
                if col_key not in warned_collision:
                    print(
                        f'[joint_converter] warning: multi channels '
                        f'mapped to "{canonical}", latter overrides '
                        f'(raw="{cam}")')
                    warned_collision.add(col_key)
            cams_merged[canonical] = cam_info

        ordered_cams = {}
        for c in WESTWELL_CAMS:
            if c in cams_merged:
                ordered_cams[c] = cams_merged[c]
        for k, v in cams_merged.items():
            if k not in ordered_cams:
                ordered_cams[k] = v
        info['cams'] = ordered_cams

        # ---- LiDAR sweeps ----
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if sd_rec['prev'] != '':
                sweep = obtain_sensor2top(
                    nusc, sd_rec['prev'], l2e_t, l2e_r_mat,
                    e2g_t, e2g_r_mat, 'lidar')
                if sweep['data_path'].endswith('.pcd'):
                    sweep['data_path'] = sweep['data_path'].replace(
                        '.pcd', '.bin')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps

        # ---- Vectorized map GT (MapTR) ----
        if map_location in nusc_maps:
            info = obtain_vectormap(
                nusc_maps, map_explorer, info, point_cloud_range,
                use_ego_origin, map_yaw_offset_deg)
        else:
            info['annotation'] = {
                'divider': [], 'ped_crossing': [],
                'boundary': [], 'centerline': [],
                'bar_markings': [], 'bar_markings_curve': [],
            }

        # ---- 3D Object Detection GT (BEVFusion) ----
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            if len(annotations) > 0 and len(boxes) > 0:
                locs = np.array(
                    [b.center for b in boxes]).reshape(-1, 3)
                dims = np.array(
                    [b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array(
                    [b.orientation.yaw_pitch_roll[0] for b in boxes]
                ).reshape(-1, 1)
                velocity = np.array(
                    [nusc.box_velocity(token)[:2]
                     for token in sample['anns']])
                valid_flag = np.array(
                    [(a['num_lidar_pts'] + a['num_radar_pts']) > 0
                     for a in annotations],
                    dtype=bool).reshape(-1)
                for i in range(len(boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = (velo
                            @ np.linalg.inv(e2g_r_mat).T
                            @ np.linalg.inv(l2e_r_mat).T)
                    velocity[i] = velo[:2]
                names = np.array([b.name for b in boxes])
                gt_boxes = np.concatenate(
                    [locs, dims, -rots - np.pi / 2], axis=1)
                assert len(gt_boxes) == len(annotations), \
                    f'{len(gt_boxes)} != {len(annotations)}'
                info['gt_boxes'] = gt_boxes
                info['gt_names'] = names
                info['gt_velocity'] = velocity.reshape(-1, 2)
                info['num_lidar_pts'] = np.array(
                    [a['num_lidar_pts'] for a in annotations])
                info['num_radar_pts'] = np.array(
                    [a['num_radar_pts'] for a in annotations])
                info['valid_flag'] = valid_flag
            else:
                info['gt_boxes'] = np.zeros((0, 7))
                info['gt_names'] = np.array([])
                info['gt_velocity'] = np.zeros((0, 2))
                info['num_lidar_pts'] = np.array([], dtype=np.int64)
                info['num_radar_pts'] = np.array([], dtype=np.int64)
                info['valid_flag'] = np.array([], dtype=bool)
        else:
            info['gt_boxes'] = np.zeros((0, 7))
            info['gt_names'] = np.array([])
            info['gt_velocity'] = np.zeros((0, 2))
            info['num_lidar_pts'] = np.array([], dtype=np.int64)
            info['num_radar_pts'] = np.array([], dtype=np.int64)
            info['valid_flag'] = np.array([], dtype=bool)

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


# ---------------------------------------------------------------------------
# Map loading helpers
# ---------------------------------------------------------------------------

def _make_map_overlay_for_known_alias(dataroot, custom_map_name):
    """Overlay dataroot to bypass NuScenesMap whitelist for custom names."""
    dataroot = osp.abspath(dataroot)
    known_aliases = [
        'singapore-onenorth', 'singapore-hollandvillage',
        'singapore-queenstown', 'boston-seaport',
    ]
    custom_json = osp.join(
        dataroot, 'maps', 'expansion', f'{custom_map_name}.json')
    if not osp.isfile(custom_json):
        return None, None
    overlay_root = tempfile.mkdtemp(
        prefix=f'nusc_map_overlay_{custom_map_name}_')
    exp_dir = osp.join(overlay_root, 'maps', 'expansion')
    base_dir = osp.join(overlay_root, 'maps', 'basemap')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    alias_name = known_aliases[0]
    alias_json = osp.join(exp_dir, f'{alias_name}.json')
    try:
        os.symlink(custom_json, alias_json)
    except Exception:
        import shutil
        shutil.copy2(custom_json, alias_json)
    custom_png = osp.join(
        dataroot, 'maps', 'basemap', f'{custom_map_name}.png')
    alias_png = osp.join(base_dir, f'{alias_name}.png')
    if osp.isfile(custom_png):
        try:
            os.symlink(custom_png, alias_png)
        except Exception:
            import shutil
            shutil.copy2(custom_png, alias_png)
    return overlay_root, alias_name


def _load_nusc_map_robust(dataroot, loc_name):
    try:
        return NuScenesMap(dataroot=dataroot, map_name=loc_name)
    except AssertionError as e:
        if 'Unknown map name' not in str(e):
            raise
        overlay_root, alias_name = _make_map_overlay_for_known_alias(
            dataroot, loc_name)
        if overlay_root is None:
            raise
        print(
            f'[joint_converter] map_name="{loc_name}" not in '
            f'nuscenes-devkit whitelist; fallback to alias '
            f'"{alias_name}" using overlay: {overlay_root}')
        return NuScenesMap(dataroot=overlay_root, map_name=alias_name)


def _discover_map_locations(nusc):
    locations = set()
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        locations.add(log['location'])
    return sorted(locations)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def create_nuscenes_infos(root_path, out_path, can_bus_root_path,
                          info_prefix, version='v1.0-trainval',
                          max_sweeps=10, maps=None,
                          point_cloud_range=None,
                          use_ego_origin=False,
                          map_yaw_offset_deg=0.0,
                          cam_channel_map=None,
                          path_substring_map=None,
                          site_name=None,
                          max_samples=None):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus

    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    try:
        nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    except Exception:
        print(f'Warning: CAN bus not found at {can_bus_root_path}, '
              f'continuing without CAN bus data')
        nusc_can_bus = None

    if maps:
        MAPS = [m.strip() for m in maps if m.strip()]
    else:
        MAPS = _discover_map_locations(nusc)
    print(f'[joint_converter] Map locations: {MAPS}')

    nusc_maps = {}
    map_explorer = {}
    for loc in MAPS:
        nusc_maps[loc] = _load_nusc_map_robust(root_path, loc)
        map_explorer[loc] = CNuScenesMapExplorer(nusc_maps[loc])

    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers, \
        f'{version} not in {available_vers}'
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(
        filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    if point_cloud_range is None:
        point_cloud_range = [-15.0, -10.0, -10.0, 15.0, 30.0, 10.0]
    patch_size, patch_center_local = get_patch_size_and_center(
        point_cloud_range)

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, nusc_maps, map_explorer,
        train_scenes, val_scenes, test,
        max_sweeps=max_sweeps,
        point_cloud_range=point_cloud_range,
        use_ego_origin=use_ego_origin,
        map_yaw_offset_deg=map_yaw_offset_deg,
        cam_channel_map=cam_channel_map,
        path_substring_map=path_substring_map,
        site_name=site_name,
        max_samples=max_samples,
    )

    metadata = dict(
        version=version,
        map_point_cloud_range=list(point_cloud_range),
        map_patch_size=list(patch_size),
        map_patch_center_offset=list(patch_center_local),
    )
    if test:
        all_infos = train_nusc_infos
        print('test sample: {}'.format(len(all_infos)))
        data = dict(infos=all_infos, metadata=metadata)
        info_test_path = osp.join(
            out_path, '{}_map_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_test_path)
        print(f'[joint_converter] Saved {len(all_infos)} infos -> {info_test_path}')
    else:
        all_infos = train_nusc_infos + val_nusc_infos
        print('train sample: {}, val sample: {}, merged: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos), len(all_infos)))
        outputs = [
            ('train', train_nusc_infos),
            ('val', val_nusc_infos),
        ]
        for split, infos in outputs:
            data = dict(infos=infos, metadata=metadata)
            info_path = osp.join(
                out_path, '{}_map_infos_temporal_{}.pkl'.format(
                    info_prefix, split))
            mmcv.dump(data, info_path)
            print(f'[joint_converter] Saved {len(infos)} infos -> {info_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    parser = argparse.ArgumentParser(
        description='Joint BEVFusion+MapTRv2 data converter for '
                    'Westwell NuScenes-like datasets')
    parser.add_argument('--root-path', type=str,
                        default='./data/nuscenes',
                        help='dataset root path')
    parser.add_argument('--canbus', type=str, default='./data',
                        help='nuScenes canbus root')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='dataset version')
    parser.add_argument('--max-sweeps', type=int, default=10)
    parser.add_argument('--maps', nargs='*', default=None,
                        help='map location names (auto-discover if omitted)')
    parser.add_argument('--point-cloud-range', nargs=6, type=float,
                        default=[-15.0, -10.0, -10.0, 15.0, 30.0, 10.0],
                        help='x_min y_min z_min x_max y_max z_max')
    parser.add_argument('--use-ego-origin', action='store_true',
                        help='use ego pose as map origin '
                             '(ignore lidar2ego extrinsic)')
    parser.add_argument('--map-yaw-offset-deg', type=float, default=0.0)
    parser.add_argument('--site-name', type=str, default=None,
                        help='site name (enables inv_flag for uk/hit)')
    parser.add_argument('--cam-remap', type=str, default=None,
                        help='JSON remap file for camera channels')
    parser.add_argument('--cam-remap-kv', nargs='*', default=None,
                        help='CLI channel map pairs: SRC=CANON')
    parser.add_argument('--nuscenes-6cam-slot-remap', action='store_true',
                        help='remap official nuScenes 6-cam to '
                             'westwell 8-slot')
    parser.add_argument('--out-dir', type=str, default='./data/nuscenes')
    parser.add_argument('--extra-tag', type=str,
                        default='nuscenes_bevfusion_maptr')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='limit converted samples for quick inference/debug')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()

    _remap_tools_dir = osp.dirname(osp.abspath(__file__))
    if _remap_tools_dir not in sys.path:
        sys.path.insert(0, _remap_tools_dir)
    from westwell_cam_remap_util import (
        load_cam_remap_json,
        merge_remap_dicts,
        nuscenes_6cam_to_westwell_channel_map,
        parse_cam_remap_kv,
    )

    _remap_cfg = merge_remap_dicts(
        load_cam_remap_json(args.cam_remap),
        parse_cam_remap_kv(args.cam_remap_kv),
    )
    _channel_map = dict(_remap_cfg.get('channel_map', {}) or {})
    if args.nuscenes_6cam_slot_remap:
        _merged = nuscenes_6cam_to_westwell_channel_map()
        _merged.update(_channel_map)
        _channel_map = _merged
    _path_sub_map = dict(_remap_cfg.get('path_substring_map', {}) or {})

    if args.version in ('v1.0-mini', 'v1.0-trainval'):
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
            cam_channel_map=_channel_map or None,
            path_substring_map=_path_sub_map or None,
            site_name=args.site_name,
            max_samples=args.max_samples,
        )
    elif args.version == 'v1.0-test':
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
            cam_channel_map=_channel_map or None,
            path_substring_map=_path_sub_map or None,
            site_name=args.site_name,
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
            cam_channel_map=_channel_map or None,
            path_substring_map=_path_sub_map or None,
            site_name=args.site_name,
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
            cam_channel_map=_channel_map or None,
            path_substring_map=_path_sub_map or None,
            site_name=args.site_name,
        )
