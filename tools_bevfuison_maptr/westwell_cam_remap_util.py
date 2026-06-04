"""
Westwell / NuScenes-like multi-camera naming: canonical 8-slot layout + remapping helpers.

Used by westwell_nusc_map_converter (pkl generation) and westwell_nusc_vis_gt_cam (debug vis).
"""

import json
import os.path as osp
from typing import Dict, List, Optional, Tuple

# Canonical 8-camera layout (same order as training vis).
CAMS = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT_MID',
    'CAM_FRONT_RIGHT',
    'CAM_RIGHT',
    'CAM_REAR_RIGHT',
    'CAM_REAR_MID',
    'CAM_REAR_LEFT',
    'CAM_LEFT',
]

# Built-in aliases: longer tokens should match before shorter ones (handled in infer_canonical_cam_name).
CAM_ALIASES = {
    'CAM_FRONT_LEFT': ['CAM_FRONT_LEFT', 'CAM_FRONT LEFT'],
    'CAM_FRONT_MID': [
        'CAM_FRONT_MID', 'CAM_MID_FRONT', 'CAM_MID FRONT', 'CAM_FRONT MID',
        'CAM_FRONT',  # nuScenes single forward cam -> mid (see ordering note below)
    ],
    'CAM_FRONT_RIGHT': ['CAM_FRONT_RIGHT', 'CAM_FRONT RIGHT'],
    'CAM_RIGHT': ['CAM_RIGHT', 'CAM RIGHT'],
    'CAM_REAR_RIGHT': ['CAM_REAR_RIGHT', 'CAM_REAR RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK RIGHT'],
    'CAM_REAR_MID': [
        'CAM_REAR_MID', 'CAM_REAR MID', 'CAM_MID_REAR', 'CAM_MID REAR',
        'CAM_BACK', 'CAM_REAR',  # nuScenes rear single / generic rear
    ],
    'CAM_REAR_LEFT': ['CAM_REAR_LEFT', 'CAM_REAR LEFT', 'CAM_BACK_LEFT', 'CAM_BACK LEFT'],
    'CAM_LEFT': ['CAM_LEFT', 'CAM LEFT'],
}


def _normalize_cam_name(text: Optional[str]) -> str:
    if text is None:
        return ''
    t = str(text).upper().replace('\\', '/')
    for ch in [' ', '-', '/', '__']:
        t = t.replace(ch, '_')
    while '__' in t:
        t = t.replace('__', '_')
    return t.strip('_')


def _build_alias_match_pairs(extra_path_substrings: Optional[Dict[str, str]] = None
                             ) -> List[Tuple[str, str]]:
    """(normalized_alias, canonical) sorted by alias length descending (longest match first)."""
    pairs: List[Tuple[str, str]] = []
    if extra_path_substrings:
        for raw_key, canonical in extra_path_substrings.items():
            ck = _normalize_cam_name(canonical)
            if ck not in CAMS:
                continue
            pairs.append((_normalize_cam_name(raw_key), ck))
    for canonical, aliases in CAM_ALIASES.items():
        for alias in aliases:
            pairs.append((_normalize_cam_name(alias), canonical))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    # Dedupe by alias string, keep first (longest / user overrides first if we prepend user - we did)
    seen = set()
    out: List[Tuple[str, str]] = []
    for a, c in pairs:
        if a in seen or not a:
            continue
        seen.add(a)
        out.append((a, c))
    return out


def load_cam_remap_json(path: Optional[str]) -> Dict:
    """Load remap JSON. Schema:

    {
      "channel_map": { "CAM_FRONT": "CAM_FRONT_MID", ... },   # nuScenes sample channel -> canonical
      "path_substring_map": { "my_front_mid": "CAM_FRONT_MID", ... }  # substring in filepath -> canonical
    }
    """
    if not path:
        return {}
    path = osp.expanduser(path)
    if not osp.isfile(path):
        raise FileNotFoundError(f'cam remap json not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('cam remap json must be a dict')
    out = {}
    cm = data.get('channel_map') or data.get('channel_map_from_nuscenes')
    if cm and isinstance(cm, dict):
        out['channel_map'] = { _normalize_cam_name(k): _normalize_cam_name(v) for k, v in cm.items() }
    psm = data.get('path_substring_map') or data.get('path_substrings')
    if psm and isinstance(psm, dict):
        out['path_substring_map'] = {
            _normalize_cam_name(k): _normalize_cam_name(v) for k, v in psm.items()
        }
    return out


def parse_cam_remap_kv(pairs: Optional[List[str]]) -> Dict:
    """Parse CLI items like CAM_FRONT=CAM_FRONT_MID into channel_map."""
    if not pairs:
        return {}
    cm = {}
    for p in pairs:
        if '=' not in p:
            raise ValueError(f'Invalid --cam-remap-kv entry (need SRC=CANON): {p}')
        a, b = p.split('=', 1)
        cm[_normalize_cam_name(a.strip())] = _normalize_cam_name(b.strip())
    return {'channel_map': cm}


def merge_remap_dicts(*parts: Optional[Dict]) -> Dict:
    merged = {'channel_map': {}, 'path_substring_map': {}}
    for d in parts:
        if not d:
            continue
        if 'channel_map' in d:
            merged['channel_map'].update(d['channel_map'])
        if 'path_substring_map' in d:
            merged['path_substring_map'].update(d['path_substring_map'])
    return merged


def remap_nuscenes_channel(
        raw_channel: str,
        channel_map: Optional[Dict[str, str]] = None,
) -> str:
    """Map raw nuScenes `sample['data']` camera key; returns normalized name or mapped canonical."""
    key = _normalize_cam_name(raw_channel)
    if channel_map and key in channel_map:
        target = channel_map[key]
        if target not in CAMS:
            raise ValueError(
                f'channel_map[{raw_channel}] -> {target} is not in canonical CAMS list'
            )
        return target
    return key


def nuscenes_6cam_to_westwell_channel_map() -> Dict[str, str]:
    """官方 nuScenes 6 路相机 channel 名 -> westwell 8 路 canonical 槽位（仅左右后三路与 mid）。"""
    return {
        'CAM_FRONT': 'CAM_FRONT_MID',
        'CAM_BACK': 'CAM_REAR_MID',
        'CAM_FRONT_LEFT': 'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT': 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT': 'CAM_REAR_LEFT',
        'CAM_BACK_RIGHT': 'CAM_REAR_RIGHT',
    }


def infer_canonical_cam_name(
        filepath: Optional[str],
        extra_path_substrings: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Infer canonical CAM_* slot from image filepath using longest-substring alias match."""
    if filepath is None:
        return None
    normalized_path = _normalize_cam_name(filepath)
    pairs = _build_alias_match_pairs(extra_path_substrings)
    for alias_norm, canonical in pairs:
        if alias_norm and alias_norm in normalized_path:
            return canonical
    return None
