
from . import _version
__version__ = _version.get_versions()['version']

from .data.datasets.load_datasets import load_data