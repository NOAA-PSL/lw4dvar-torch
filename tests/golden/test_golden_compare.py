"""Unit tests for the golden machinery itself: the comparator
(`golden_compare.compare`) and the mini-ACE2 tolerance router
(`mini_ace2.tolerance`), so a change that loosens either is caught even
while the frozen baselines still pass. CPU only; no model runs."""

import os
import sys

import pytest

from golden_compare import compare

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ace2'))
import mini_ace2  # noqa: E402


@pytest.mark.parametrize('path, expected', [
    ('/config/exp/bg_check', (0.0, 0.0)),
    ('/obs/background/innov_mean[0]', (1e-5, 1e-4)),
    ('/obs/background/innov_rms[0]', (1e-5, 0.0)),
    ('/obs/background/used[3]', (1e-5, 0.0)),
    ('/obs/available[0]', (1e-5, 0.0)),
    ('/obs/qc_flag_counts/8[2]', (1e-5, 0.0)),
    ('/jo_per_slot_first[4]', (1e-5, 0.0)),
    ('/loss[0]', (1e-4, 0.0)),
    ('/loss[4]', (1e-4, 0.0)),     # epoch 5: the last tight one
    ('/loss[5]', (1e-2, 0.0)),     # epoch 6
    ('/loss[49]', (1e-2, 0.0)),
    ('/obs/analysis/used[2]', (0.0, 5)),
    ('/obs/analysis/innov_mean[1]', (0.0, 0.01)),
    ('/obs/analysis/innov_rms[1]', (1e-2, 0.0)),
    ('/analysis_forecast/h500/mean[0]', (1e-3, 0.0)),
    ('/jo_per_slot_last[2]', (1e-2, 0.0)),
])
def test_mini_ace2_tolerance_routing(path, expected):
    assert mini_ace2.TIGHT_EPOCHS == 5
    assert mini_ace2.tolerance(path) == expected


def test_compare_callable_absolute_tolerance():
    def tol(path):
        return (0.0, 0.5) if path.startswith('/a') else (0.0, 0.0)

    assert not compare({'a': [10.0], 'b': [1.0]}, {'a': [10.4], 'b': [1.0]}, tol)
    assert compare({'a': [10.0]}, {'a': [10.6]}, tol) == [('/a[0]', 10.0, 10.6)]
    assert compare({'b': [1]}, {'b': [2]}, tol) == [('/b[0]', 1, 2)]   # counts exact without atol


def test_compare_relative_tolerance():
    assert not compare({'x': 100.0}, {'x': 100.0 + 1e-7}, 1e-8)  # 1e-9 relative
    assert compare({'x': 100.0}, {'x': 100.0 + 1e-5}, 1e-8)


def test_compare_structure_mismatches():
    assert compare({'a': 1, 'b': 2}, {'a': 1}, 0.0) == [('/b', 2, '<missing>')]
    assert compare({'a': 1}, {'a': 1, 'c': 3}, 0.0) == [('/c', '<missing>', 3)]
    assert compare({'a': [1, 2, 3]}, {'a': [1, 2]}, 0.0) == [('/a (length)', 3, 2)]
    assert compare({'a': [1.0]}, {'a': 1.0}, 0.0) == [('/a', [1.0], 1.0)]          # list vs number
    assert compare({'a': {'x': 1}}, {'a': [1]}, 0.0) == [('/a', {'x': 1}, [1])]    # dict vs list
    assert compare({'a': True}, {'a': 1}, 1.0) == [('/a', True, 1)]                 # bool is not a number
    assert compare({'a': None}, {'a': 0.0}, 1.0) == [('/a', None, 0.0)]
    assert compare({'s': 'logpinterp'}, {'s': 'ps'}, 1.0) == [('/s', 'logpinterp', 'ps')]
    assert not compare({'a': {'b': [1, {'c': 'x'}]}}, {'a': {'b': [1, {'c': 'x'}]}}, 0.0)
