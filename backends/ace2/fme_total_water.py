"""
Total water from its components, for ACE2 checkpoints trained on the CRTM
layer families (ace2_dataset.py --layer_families crtm), which carry
specific_humidity_<k> and specific_total_condensate_<k> but no
specific_total_water_<k>.

fme 2026.5 finds total water only by the `specific_total_water_` prefix,
and its correctors (conserve_dry_air, moisture_budget_correction) read it,
so without this a CRTM checkpoint's Stepper.step raises KeyError. `install()`
makes AtmosphereData.specific_total_water fall back to the sum of the
components when the total is absent; data that carry the total are
unaffected. Same patch as training (backends/ace2/training/fme_train.py,
see ace2-training.md "Total water for the corrector"), so inference applies
the corrector exactly as the checkpoint was trained with.
"""

from fme.core.atmosphere_data import AtmosphereData
from fme.core.stacker import Stacker

_components = Stacker(
    {
        "specific_humidity": ["specific_humidity_"],
        "specific_total_condensate": ["specific_total_condensate_"],
    }
)


def install() -> None:
    """Patch AtmosphereData.specific_total_water (idempotent)."""
    prop = AtmosphereData.specific_total_water
    if getattr(prop.fget, "_total_water_fallback", False):
        return
    stacked = prop.fget

    def _specific_total_water(self):
        try:
            return stacked(self)
        except KeyError:
            try:
                return _components("specific_humidity", self.data) + _components(
                    "specific_total_condensate", self.data
                )
            except KeyError:
                pass
            raise

    _specific_total_water._total_water_fallback = True
    AtmosphereData.specific_total_water = property(_specific_total_water, doc=prop.__doc__)
