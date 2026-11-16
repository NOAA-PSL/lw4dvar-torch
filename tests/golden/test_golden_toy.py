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
