"""CLI for the adjoint-quality battery (notes/adjoint_quality.md). Run from the repo root:

GPU (model env):
  python -m diagnostics.adjoint_quality.aq all       --config CFG --out DIR [--golden PAT] [--golden_cv PAT] [--smoke]
  python -m diagnostics.adjoint_quality.aq gradient  --config CFG --out DIR ...      (parts can also run separately;
  python -m diagnostics.adjoint_quality.aq survival  --config CFG --out DIR ...       they merge into one JSON per date)
  python -m diagnostics.adjoint_quality.aq single_ob --config CFG --out DIR --loc sh nh [--b_cov NPZ] [--smoke]
  python -m diagnostics.adjoint_quality.aq families  --config CFG --out DIR [--sizes 0.02 0.005] [--smoke]
CPU (anemoi-dev env for the maps):
  python -m diagnostics.adjoint_quality.aq plot    --results DIR [--single_ob DIR] --figs DIR [--label NAME]
  python -m diagnostics.adjoint_quality.aq compare --models NAME=DIR_OR_BUNDLE ... [--no_reference] --figs DIR
  python -m diagnostics.adjoint_quality.aq bundle  --results DIR --single_ob_metrics JSON --label NAME --out FILE

Settings: defaults in battery.DEFAULTS; --settings YAML overrides them; explicit flags override both.
GOLDEN patterns may contain {mmdd} and {date}, e.g. 'output/pita22_ot_s_{mmdd}/*_latent_increment_120h_200it.pt'.
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference', 'a2_e065_pita22.json')


def _settings(a):
    from .battery import DEFAULTS, SMOKE
    s = dict(DEFAULTS)
    if a.settings:
        import yaml
        s.update(yaml.safe_load(open(a.settings)) or {})
    if a.smoke:
        s.update(SMOKE)
    for k in ('leads', 'eps', 'stab'):
        if getattr(a, k) is not None:
            s[k] = tuple(getattr(a, k))
    for k in ('svd_rank', 'svd_steps', 'target_resp6'):
        if getattr(a, k) is not None:
            s[k] = getattr(a, k)
    s['smoke'] = bool(a.smoke)
    return s


def _merge_write(path, meta, R):
    old = json.load(open(path)) if os.path.exists(path) else {}
    if old and old.get('date') != R.get('date'):
        raise RuntimeError(f'{path} holds another date ({old.get("date")})')
    parts = sorted(set(old.get('meta', {}).get('parts', [])) | set(meta.pop('parts')))
    out = {**old, **R, 'meta': {**meta, 'parts': parts}, 'smoke': meta['settings']['smoke']}
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('saved', path, flush=True)


def cmd_battery(a):
    from .battery import Battery, Setup
    s = _settings(a)
    S = Setup(a.config, tf32=a.tf32)
    B = Battery(S, s, golden=a.golden, golden_cv=a.golden_cv)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f'{a.tag}_{S.mmdd}' + ('_smoke' if a.smoke else ''))
    parts = ['gradient', 'survival'] if a.cmd == 'all' else [a.cmd]
    if 'gradient' in parts:
        B.gradient()
    if 'survival' in parts:
        B.survival(base)
    meta = S.meta(a.cmd, s)
    meta.update(parts=parts, golden=a.golden, golden_cv=a.golden_cv)
    _merge_write(base + '.json', meta, B.R)
    S.log('done')


def cmd_single_ob(a):
    from . import single_ob
    from .battery import Setup
    S = Setup(a.config, tf32=a.tf32)
    obs = ('a',) if a.smoke else tuple(a.obs)
    iters = 10 if a.smoke else a.iters
    for loc in a.loc:
        af = yaml.safe_load(a.adjoint_filter) if a.adjoint_filter else None
        res = single_ob.run(S, a.out, loc, obs=obs, iters=iters, b_cov=a.b_cov, sigma_b=a.sigma_b, adjoint_filter=af)
        res['meta'] = S.meta('single_ob', {'loc': loc, 'obs': list(obs), 'iters': iters, 'b_cov': a.b_cov,
                                           'sigma_b': a.sigma_b, 'smoke': bool(a.smoke),
                                           'adjoint_filter': af})
        with open(os.path.join(a.out, f'{S.mmdd}_{res["loc"]}_summary.json'), 'w') as fh:
            json.dump(res, fh, indent=1)
    S.log('done')


def cmd_families(a):
    from . import families
    from .battery import Setup
    S = Setup(a.config, tf32=a.tf32)
    sizes = (0.004, 0.001) if a.smoke else tuple(a.sizes)
    R = families.run(S, sizes=sizes, leads_h=(6, 24) if a.smoke else (6, 24, 48), n_real=1 if a.smoke else 2)
    R['date'] = S.date
    R['meta'] = S.meta('families', {'sizes': list(sizes), 'smoke': bool(a.smoke)})
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f'families_{S.mmdd}' + ('_smoke' if a.smoke else '') + '.json')
    with open(path, 'w') as fh:
        json.dump(R, fh, indent=1)
    print('saved', path, flush=True)
    S.log('done')


def cmd_plot(a):
    from . import plots
    if a.results:
        R, _ = plots.load_dir(a.results)
        if R:
            plots.plot_aq(R, a.figs, a.label)
    if a.single_ob:
        plots.plot_single_ob(a.single_ob, a.figs, a.label)


def cmd_compare(a):
    from . import plots
    models = []
    if not a.no_reference:
        R, so = plots.load_dir(REFERENCE)
        models.append((json.load(open(REFERENCE))['meta']['label'], R, so))
    for m in a.models:
        label, path = m.split('=', 1)
        R, so = plots.load_dir(path)
        models.append((label, R, so))
    plots.compare(models, a.figs)


def cmd_bundle(a):
    from . import plots
    R, _ = plots.load_dir(a.results)
    for r in R:
        r.pop('_maps_file', None)
    so = json.load(open(a.single_ob_metrics)) if a.single_ob_metrics else None
    meta = {'label': a.label, 'note': a.note, 'n_dates': len(R), 'dates': [r['date'] for r in R]}
    with open(a.out, 'w') as fh:
        json.dump({'meta': meta, 'aq': R, 'single_ob': so}, fh, separators=(',', ':'))
    print('saved', a.out, os.path.getsize(a.out), 'bytes')


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m diagnostics.adjoint_quality.aq', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('all', 'gradient', 'survival', 'single_ob'):
        q = sub.add_parser(name)
        q.add_argument('--config', required=True, help='4D-Var config (background = its first guess, twin from its sdate)')
        q.add_argument('--out', required=True, help='output directory')
        q.add_argument('--smoke', action='store_true', help='tiny settings for a quick end-to-end check')
        q.add_argument('--tf32', action='store_true', help='allow TF32 matmuls in "fp32" (default: true fp32)')
        if name == 'single_ob':
            q.add_argument('--loc', nargs='+', default=['sh', 'nh'], help="'sh' (50S 90E), 'nh' (45N 180E) or 'lat,lon'")
            q.add_argument('--obs', nargs='+', default=['a', 'b', 'c'], choices=['a', 'b', 'c'])
            q.add_argument('--iters', type=int, default=200)
            q.add_argument('--b_cov', default=None, help='latent_eof class covariance npz for the BG solver (optional)')
            q.add_argument('--adjoint_filter', default=None,
                           help="exp.adjoint_filter as YAML, e.g. '{diffusion: {scale_km: 300, steps: 1}}'")
            q.add_argument('--sigma_b', type=float, default=0.07)
        else:
            q.add_argument('--tag', default='aq', help='output file prefix: {tag}_{mmdd}.json')
            q.add_argument('--golden', default=None, help='latent_increment .pt pattern of a 4D-Var run (twin) for this model')
            q.add_argument('--golden_cv', default=None, help='same, real-obs run (optional)')
            q.add_argument('--settings', default=None, help='YAML overriding battery.DEFAULTS')
            q.add_argument('--leads', type=int, nargs='+', default=None, help='cost leads in hours (default 24 120)')
            q.add_argument('--eps', type=float, nargs='+', default=None, help='Taylor steps / |u_ref|')
            q.add_argument('--stab', type=float, nargs='+', default=None, help='stability moves / |u_ref|')
            q.add_argument('--svd_rank', type=int, default=None)
            q.add_argument('--svd_steps', type=int, default=None, help='model steps after 6 h (7 -> 48 h)')
            q.add_argument('--target_resp6', type=float, default=None, help='6-h response for |u_ref| without --golden')
    q = sub.add_parser('families')
    q.add_argument('--config', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--smoke', action='store_true')
    q.add_argument('--tf32', action='store_true')
    q.add_argument('--sizes', type=float, nargs='+', default=[0.02, 0.005],
                   help='perturbation sizes in the normalised all-prognostic norm (first = main, second = linear check)')
    q = sub.add_parser('plot')
    q.add_argument('--results', default=None)
    q.add_argument('--single_ob', default=None)
    q.add_argument('--figs', required=True)
    q.add_argument('--label', default='model')
    q = sub.add_parser('compare')
    q.add_argument('--models', nargs='*', default=[], help='NAME=results_dir_or_bundle')
    q.add_argument('--no_reference', action='store_true', help='do not include the A2 reference bundle')
    q.add_argument('--figs', required=True)
    q = sub.add_parser('bundle')
    q.add_argument('--results', required=True)
    q.add_argument('--single_ob_metrics', default=None)
    q.add_argument('--label', required=True)
    q.add_argument('--note', default='')
    q.add_argument('--out', required=True)
    a = p.parse_args(argv)
    {'families': cmd_families, 'all': cmd_battery, 'gradient': cmd_battery, 'survival': cmd_battery, 'single_ob': cmd_single_ob,
     'plot': cmd_plot, 'compare': cmd_compare, 'bundle': cmd_bundle}[a.cmd](a)


if __name__ == '__main__':
    main()
