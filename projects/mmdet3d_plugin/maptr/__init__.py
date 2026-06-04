from .assigners import *
from .dense_heads import *
from .modules import *
from .losses import *

# MapTR / MapTRv2 完整检测器按需加载（独立 MapTR 训练配置使用）
_DETECTORS = {
    'MapTR': ('.detectors.maptr', 'MapTR'),
    'MapTRv2': ('.detectors.maptrv2', 'MapTRv2'),
}


def __getattr__(name):
    if name in _DETECTORS:
        import importlib
        module_path, attr = _DETECTORS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
