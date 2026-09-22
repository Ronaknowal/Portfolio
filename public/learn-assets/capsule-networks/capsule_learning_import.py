"""Load the hyphenated downloadable teaching program without executing its main."""
from pathlib import Path
import importlib.util


def load_learning():
    source = Path(__file__).resolve().parent / "capsule-learning.py"
    spec = importlib.util.spec_from_file_location("capsule_learning", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
