"""Forward-only diagnostic: does injecting the saved (bad) latent increment
from the 16-step Aurora collapse actually produce NaN/Inf in the decoded
state, or finite-but-extreme values that merely fail QC?

Loads the real IC/background used by config_aurora_bisect_16.yml (2014-12-30T00
init, 48h back-forecast to 2015-01-01T00), reuses the "best" saved increment
from that same run's output/ directory (test_aurora_bisect_16, the 3-epoch
run whose epoch-2 increment produced the total-QC-rejection collapse), and
advances the model 16 steps forward under torch.no_grad(), printing
isnan/isinf flags and min/max for every decoded field at every step -- both
the core prognostic fields and the unclipped output-only diagnostic
surface fields (scaled_tp_1h, ssrd_1h, ttr_1h, uvb_1h, i10fg, blh), which
are the most likely place an fp16-autocast overflow would first appear
(zero-initialized, never clamped by apply_rollout_input_clipping).
"""
import os
import sys
import torch

# This script lives in backends/aurora/, but long_window_4dvar_utils.py
# lives at the repo root -- see CLAUDE.md "Login-node ERA5 prefetch
# scripts, and a sys.path bug they share" for the same issue found in the
# prefetch scripts; same fix here.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import long_window_4dvar_utils as u

CONFIG = 'config_aurora_bisect_16.yml'
INCREMENT_PATH = 'output/test_aurora_bisect_16/2015-01-01T00_latent_increment_96h_3it.pt'

_CORE = ('z', 'u', 'v', 't', 'q', 'sp')
_UNCLIPPED_DIAG = ('i10fg', 'blh', 'uvb_1h', 'ssrd_1h', 'ttr_1h', 'scaled_tp_1h', 'scaled_sf_1h')


def report(tag, decoded):
    bad_any = False
    for name, val in decoded.items():
        if not torch.is_tensor(val):
            continue
        nnan = torch.isnan(val).sum().item()
        ninf = torch.isinf(val).sum().item()
        if nnan or ninf:
            bad_any = True
            print(f"  [{tag}] {name}: NaN={nnan} Inf={ninf} shape={tuple(val.shape)}")
        elif name in _CORE or name in _UNCLIPPED_DIAG:
            print(f"  [{tag}] {name}: min={val.min().item():.4g} max={val.max().item():.4g}")
    if not bad_any:
        print(f"  [{tag}] no NaN/Inf in any decoded field")
    return bad_any


def main():
    exp, windows = u.load_config(CONFIG)
    logger = u.get_logger()
    model = u.get_model(exp)
    exp['date'] = exp['sdate']
    exp['restart_window'] = exp['restart']
    window_name = next(iter(windows.keys()))
    exp = u.get_window(exp, window_name, windows)
    input_encoded, exp = u.get_input(exp, model, logger)

    increment = torch.load(INCREMENT_PATH, map_location=model.device)
    if isinstance(increment, dict):
        # in case it was saved as a state dict / wrapped tensor
        increment = next(iter(increment.values()))
    increment = increment.to(model.device)
    print(f"loaded increment: shape={tuple(increment.shape)}, "
          f"norm={increment.norm().item():.6g}, "
          f"max_abs={increment.abs().max().item():.6g}")

    model._prime_noise() if hasattr(model, '_prime_noise') else None

    with torch.no_grad():
        decoded0 = model.decode_state(input_encoded)
        print("=== step 0 (uncorrected background) ===")
        report('step0', decoded0)

        state = input_encoded
        for step in range(1, 17):
            inj = increment if step == 1 else None
            state = model.advance(state, steps=1, use_checkpoint=False, latent_increment=inj)
            decoded = model.decode_state(state)
            print(f"=== step {step} ===")
            bad = report(f'step{step}', decoded)
            if bad:
                print(f"*** first NaN/Inf appears at step {step} ***")
                sys.exit(0)

    print("no NaN/Inf found through 16 steps -- collapse is NOT NaN-driven; "
          "must be finite-but-extreme values failing QC (gross_check or "
          "interpolation bounds) -- needs a different probe.")


if __name__ == '__main__':
    main()
