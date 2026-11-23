"""Golden outputs on the toy model (layer 4, issue #20): each config's
minimization must reproduce tests/data/golden/toy_<config>.json to RTOL.
See toy_golden.py; regenerate with `python tests/make_fixtures.py golden`
and explain what moved in the PR."""

import copy
import json
import logging
import tempfile

import pytest
import xarray as xr

import toy_golden


@pytest.mark.parametrize('name', toy_golden.CONFIGS)
def test_toy_golden(name):
    with open(toy_golden.golden_path(name)) as f:
        golden = json.load(f)
    mismatches = toy_golden.compare(golden, toy_golden.run(name), toy_golden.RTOL)
    assert not mismatches, toy_golden.report(mismatches)


@pytest.mark.parametrize('change', [
    {'learn_rate': 0.0505},     # a 1% change of a default
    {'bg_check': 3.5},          # a QC threshold
])
def test_planted_change_is_caught(monkeypatch, change):
    """The comparison is not vacuous: a small change to a setting moves the
    golden numbers far outside RTOL."""
    with open(toy_golden.golden_path('default')) as f:
        golden = json.load(f)
    monkeypatch.setitem(toy_golden.CONFIGS, 'default', change)
    mismatches = toy_golden.compare(golden, toy_golden.run('default'), toy_golden.RTOL)
    losses = [m for m in mismatches if m[0].startswith('/loss[')]
    assert losses, 'a changed setting left the cost history unchanged'
    assert max(abs(g - v) / max(abs(g), abs(v)) for _, g, v in losses) > 1e3 * toy_golden.RTOL


def test_planted_operator_bias_is_caught(monkeypatch):
    """The obs are frozen, not rebuilt from the forward operator, so a
    constant operator bias (0.5 hPa) moves the innovations and the cost."""
    with open(toy_golden.golden_path('default')) as f:
        golden = json.load(f)
    orig = toy_golden.utils.get_surface_pressure
    # get_surface_pressure works in the levels' units: hPa for the toy model
    monkeypatch.setattr(toy_golden.utils, 'get_surface_pressure', lambda *a, **k: orig(*a, **k) + 0.5)
    mismatches = toy_golden.compare(golden, toy_golden.run('default'), toy_golden.RTOL)
    assert any(m[0].startswith('/loss[') for m in mismatches), 'an operator bias left the cost unchanged'


@pytest.mark.parametrize('golden, got, match', [
    (1.0, 1.0 + 1e-12, True),
    (1.0, float('inf'), False),
    (float('inf'), float('inf'), True),
    (float('inf'), float('-inf'), False),
    (float('nan'), float('nan'), True),
    (1.0, float('nan'), False),
])
def test_compare_non_finite(golden, got, match):
    assert (not toy_golden.compare({'x': [golden]}, {'x': [got]}, 1e-8)) == match


def test_save_trajectory_diagnostics_reuses_background_pass(monkeypatch):
    # save_trajectory_diagnostics must reuse compute_optimal's own
    # background QC pass (psobs_traj['background_hx'], added so the
    # optimization can freeze its gross check against the background --
    # PR #29's review) instead of recomputing it -- confirms the new
    # plumbing doesn't just avoid a crash but produces the SAME
    # background diagnostics the optimization itself used, exactly
    # matching this file's own 'quality_control_reference': 'background'
    # claim. Not exercised by the golden-value tests above (they never
    # call save_trajectory_diagnostics). save_xr_trajectory (the model-
    # state netCDF, unrelated to this check) is stubbed out: ToyModel has
    # no `_levels_by_base`, a real backend's own attribute it never
    # needed before now.
    monkeypatch.setattr(toy_golden.utils, 'save_xr_trajectory', lambda *a, **k: None)
    p = toy_golden._problem('default')
    exp = copy.deepcopy(p.exp)
    exp['max_epoch'] = 2  # fast -- this test only checks the plumbing, not convergence
    # save_trajectory_diagnostics (unlike compute_optimal) parses exp['date']
    # with add_hours/to_datetime's ISO format -- BASE's '2015010100' is fine
    # for compute_optimal (never parsed as a date there) but not here.
    exp['date'] = '2015-01-01T00'
    with tempfile.TemporaryDirectory() as tmp:
        exp['path_output'] = tmp + '/'
        analysis, _ = toy_golden.utils.compute_optimal(
            exp, p.model, p.state, p.verif_ic, p.psobs_traj, p.grid_interp, logging.getLogger('test'))
        assert 'background_hx' in p.psobs_traj
        cached_used = p.psobs_traj['background_hx']['used'].copy()
        ofile1, _, _ = toy_golden.utils.save_trajectory_diagnostics(
            exp, p.model, p.state, analysis, p.verif_ic, p.psobs_traj, 'all', p.grid_interp,
            logging.getLogger('test'),
        )
        with xr.open_dataset(ofile1) as ds:
            saved_used = ds['surface_pressure_used'].values.astype(bool)
    assert (saved_used == cached_used).all()


def test_planted_analysis_permutation_is_caught(monkeypatch):
    """A longitude shift of the returned analysis keeps its channel means
    and stds; the frozen spatial sample must catch it."""
    with open(toy_golden.golden_path('default')) as f:
        golden = json.load(f)
    orig = toy_golden.utils.compute_optimal

    def shifted(*args, **kwargs):
        analysis, lsave = orig(*args, **kwargs)
        return type(analysis)(analysis.state.roll(1, dims=-1), analysis.date), lsave

    monkeypatch.setattr(toy_golden.utils, 'compute_optimal', shifted)
    mismatches = toy_golden.compare(golden, toy_golden.run('default'), toy_golden.RTOL)
    paths = {m[0].split('[')[0] for m in mismatches}
    assert paths == {'/analysis_sample'}, paths
