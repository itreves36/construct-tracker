"""construct_tracker measures constructs in text."""

from construct_tracker.data.datasets.load_datasets import load_data as _load_data

from . import _version

__version__ = _version.get_versions()["version"]
__all__ = ["_load_data"]
