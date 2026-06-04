"""Dataset package with lazy imports to avoid loading mmdet3d before plugin registries."""

from .builder import custom_build_dataset

__all__ = [
    'custom_build_dataset',
    'CustomNuScenesDataset',
    'CustomNuScenesLocalMapDataset',
    'CustomNuScenesOfflineLocalMapDataset',
    'BEVFusionMapTRJointDataset',
]

_LAZY_ATTRS = {
    'CustomNuScenesDataset': ('.nuscenes_dataset', 'CustomNuScenesDataset'),
    'CustomNuScenesLocalMapDataset': (
        '.nuscenes_map_dataset', 'CustomNuScenesLocalMapDataset'),
    'CustomNuScenesOfflineLocalMapDataset': (
        '.nuscenes_offlinemap_dataset', 'CustomNuScenesOfflineLocalMapDataset'),
    'BEVFusionMapTRJointDataset': (
        'projects.mmdet3d_plugin.bevfusion_maptr.datasets',
        'BEVFusionMapTRJointDataset'),
}


def __getattr__(name):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    import importlib
    module_path, attr = _LAZY_ATTRS[name]
    if module_path.startswith('projects.'):
        module = importlib.import_module(module_path)
    else:
        module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
