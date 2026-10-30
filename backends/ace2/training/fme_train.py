"""`python -m fme.ace.train` with fme's SFNO gradient checkpointing made
non-reentrant, and total water computed from vapor + condensate when the data
carry those instead (see below). Run it through torchrun exactly like
fme.ace.train.

fme 2026.5.1's SFNO builders (`checkpointing: 1|3`) call
torch.utils.checkpoint.checkpoint without `use_reentrant`, which in torch 2.x
means the reentrant implementation. Checkpointing the encoder then cuts its
weights out of the graph (its input, the data, does not require grad), and
DDP fails with "Expected to have finished reduction in the prior iteration
... Parameter indices which did not receive grad: 1 2 3". Non-reentrant
checkpointing (what fme's own OptimizationConfig.checkpoint uses) computes
the same gradients. Reported as github.com/ai2cm/ace/issues/1577 (ace main
fixed only conditional_sfno, in #1493; the modulus SFNO behind the
`SphericalFourierNeuralOperatorNet` builder was still unfixed at 24cdac446).
Drop this wrapper once the env's fme has the fix. See ace2-training.md.
"""

import functools
import runpy

import torch.utils.checkpoint

_non_reentrant = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)

import fme.ace.models.modulus.layers  # noqa: E402
import fme.ace.models.modulus.sfnonet  # noqa: E402
import fme.core.models.conditional_sfno.layers  # noqa: E402
import fme.core.models.conditional_sfno.sfnonet  # noqa: E402

for _m in (
    fme.ace.models.modulus.layers,
    fme.ace.models.modulus.sfnonet,
    fme.core.models.conditional_sfno.layers,
    fme.core.models.conditional_sfno.sfnonet,
):
    _m.checkpoint = _non_reentrant

# Total water from its components, for the CRTM layer families
# (ace2_dataset.py --layer_families crtm): specific_total_water_<k> =
# specific_humidity_<k> + specific_total_condensate_<k>. fme finds total
# water only by the specific_total_water_ prefix; its correctors
# (conserve_dry_air, moisture_budget_correction) and derived total water
# path only read it, so falling back to the sum when the total is absent is
# all they need. Data that carry the total are unaffected.
from fme.core.atmosphere_data import AtmosphereData  # noqa: E402
from fme.core.stacker import Stacker  # noqa: E402

_stacked_total_water = AtmosphereData.specific_total_water.fget
_components = Stacker(
    {
        "specific_humidity": ["specific_humidity_"],
        "specific_total_condensate": ["specific_total_condensate_"],
    }
)


def _specific_total_water(self):
    try:
        return _stacked_total_water(self)
    except KeyError:
        try:
            return _components("specific_humidity", self.data) + _components(
                "specific_total_condensate", self.data
            )
        except KeyError:
            pass
        raise


AtmosphereData.specific_total_water = property(
    _specific_total_water, doc=AtmosphereData.specific_total_water.__doc__
)

if __name__ == "__main__":
    runpy.run_module("fme.ace.train", run_name="__main__", alter_sys=True)
