"""Smoke run of the full solver with a real (tiny) model: the mini ACE2
(2.5 deg, 16-layer CRTM+land variables, SFNO embed_dim 64) on CPU, one 24 h
window of ps obs, 50 epochs, through src/long_window_4dvar.py exactly as a user
runs it (tests/ace2/mini_ace2.py).

Inputs are tests/data/mini_ace2 (tests/make_fixtures.py mini_ace2) plus the
checkpoint, a GitHub release asset pinned by sha256 in that directory's
manifest.json: found at $LW4DVAR_MINI_ACE2_CKPT, else in the cache directory
($LW4DVAR_FIXTURE_CACHE, default ~/.cache/lw4dvar), else downloaded there.

Needs fme (envs/requirements-ci-ace2.txt); run with `make check-ace2`. It checks
that the code runs end to end, that the minimization works (a large, steady
cost drop), and that the outputs match the golden summary
tests/data/golden/mini_ace2.json (layer 4, issue #20); not analysis quality:
this model is far too small for that (its z500 analysis gets slightly worse).
"""

import json
import math
import os
import sys

import pytest
import xarray as xr

pytest.importorskip('fme')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'golden'))
import mini_ace2  # noqa: E402
from golden_compare import compare, report  # noqa: E402

pytestmark = [pytest.mark.ace2, pytest.mark.slow]

# observed: 2721 -> 1255 (-54%, synthetic parquet psobs+raobs fixture, raobs
# t/u/v sampled from the real background + noise -- see
# tests/make_fixtures.py's _ace2_background_profiles; -54%, not the
# pre-frozen-background-QC -46%, since PR #29's review -- the gross
# check no longer loosens as the state moves away from the background).
# The margin absorbs
# CPU/BLAS differences
MAX_FINAL_TO_FIRST = 0.6


@pytest.fixture(scope='module')
def checkpoint():
    return mini_ace2.checkpoint_path()


def test_ace2_mini_window(tmp_path, checkpoint):
    rc, log, exp, windows, out = mini_ace2.run(tmp_path, checkpoint)
    assert rc == 0, log[-4000:]

    losses = mini_ace2.epoch_losses(log)
    epochs = mini_ace2.EPOCHS
    assert sorted(losses) == list(range(1, epochs + 1)), f'epochs logged: {sorted(losses)}'
    assert all(math.isfinite(v) for v in losses.values()), losses
    assert losses[epochs] <= MAX_FINAL_TO_FIRST * losses[1], (
        f'cost fell only {losses[1]:.0f} -> {losses[epochs]:.0f} in {epochs} epochs')

    stem = f"{exp['sdate']}_"
    raobs_hits = None
    for kind in (
        'latent_increment', 'observation_diagnostics', 'control_forecast', 'optimal_forecast',
        'raobs_diagnostics',
    ):
        hits = [f for f in os.listdir(out) if f.startswith(stem + kind)]
        assert hits, f'no {kind} output in {out}: {sorted(os.listdir(out))}'
        if kind == 'raobs_diagnostics':
            raobs_hits = hits

    # The aggregate cost-drop check above can pass on psobs alone even if
    # every radiosonde profile is silently dropped (e.g. masked out
    # entirely by QC) -- check the diagnostics themselves record used
    # observations for every raobs variable this config requests, not
    # just that the file exists (PR #29's review).
    with xr.open_dataset(os.path.join(out, raobs_hits[0])) as ds:
        for base in ('t', 'u', 'v'):
            assert int(ds[base + '_used'].sum()) > 0, f'no {base} raobs observations used'

    with open(mini_ace2.GOLDEN) as f:
        golden = json.load(f)
    mismatches = compare(golden, mini_ace2.summarize(exp, windows, out), mini_ace2.tolerance)
    assert not mismatches, report(mismatches)
