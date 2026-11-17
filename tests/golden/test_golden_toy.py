"""Golden outputs on the toy model (layer 4, issue #20): each config's
minimization must reproduce tests/data/golden/toy_<config>.json to RTOL.
See toy_golden.py; regenerate with `python tests/make_fixtures.py golden`
and explain what moved in the PR."""

import json

import pytest

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
