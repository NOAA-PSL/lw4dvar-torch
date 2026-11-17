"""Put src/ on sys.path so tests import the solver modules as the driver
does (`import long_window_4dvar_utils as utils`), and the repository root so
they can import backend modules as a package (`from backends import
jax_bridge`) however pytest is invoked -- `python -m pytest` from the root
adds it, a bare `pytest` or a run from another directory does not."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
for path in (REPO_ROOT, SRC_DIR):   # src/ ends up first
    if path not in sys.path:
        sys.path.insert(0, path)
