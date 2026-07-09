"""Auto-import every migration module so its `@migration` decorator registers."""
import importlib
import pkgutil

from .base import REGISTRY, Migration, migration   # noqa: F401

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name != "base":
        importlib.import_module(f"{__name__}.{_mod.name}")
