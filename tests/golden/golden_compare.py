"""Comparison of a run's golden summary (a JSON-like tree) with the
frozen one under tests/data/golden, and the failure report."""

import math


def compare(golden, got, tol, path=''):
    """Mismatches between two JSON-like trees, as (path, golden, got) tuples.

    `tol` is a relative tolerance for every number, or a function of the
    value's path (e.g. '/loss[3]') returning `(rtol, atol)`. Two numbers
    match if |golden - got| <= max(atol, rtol * max(|golden|, |got|)), so
    integers (obs counts) match exactly unless atol >= 1. Strings, booleans,
    None and the tree's structure must match exactly."""
    out = []
    if isinstance(golden, dict) and isinstance(got, dict):
        for k in sorted(set(golden) | set(got)):
            if k not in golden or k not in got:
                out.append((f'{path}/{k}', golden.get(k, '<missing>'), got.get(k, '<missing>')))
            else:
                out += compare(golden[k], got[k], tol, f'{path}/{k}')
    elif isinstance(golden, list) and isinstance(got, list):
        if len(golden) != len(got):
            out.append((f'{path} (length)', len(golden), len(got)))
        else:
            for i, (g, v) in enumerate(zip(golden, got)):
                out += compare(g, v, tol, f'{path}[{i}]')
    elif _number(golden) and _number(got):
        rtol, atol = tol(path) if callable(tol) else (tol, 0.0)
        if math.isnan(golden) != math.isnan(got) or \
                abs(golden - got) > max(atol, rtol * max(abs(golden), abs(got))):
            out.append((path, golden, got))
    elif golden != got or type(golden) is not type(got):
        out.append((path, golden, got))
    return out


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def report(mismatches, limit=15):
    lines = [f'{len(mismatches)} value(s) differ from the golden output '
             '(regenerate with `python tests/make_fixtures.py golden` if the change is intended, '
             'and explain it in the PR):']
    for path, g, v in mismatches[:limit]:
        rel = ''
        if _number(g) and _number(v) and max(abs(g), abs(v)) > 0:
            rel = f'  (rel {abs(g - v) / max(abs(g), abs(v)):.2e})'
        lines.append(f'  {path}: golden {g!r}, got {v!r}{rel}')
    if len(mismatches) > limit:
        lines.append(f'  ... and {len(mismatches) - limit} more')
    return '\n'.join(lines)
