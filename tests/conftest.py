"""Put the repo root on sys.path so tests import the solver modules as the
driver does (`import long_window_4dvar_utils as utils`)."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
