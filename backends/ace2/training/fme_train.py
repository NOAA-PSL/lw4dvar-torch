"""`python -m fme.ace.train` with fme's SFNO gradient checkpointing made
non-reentrant (run it through torchrun exactly like fme.ace.train).

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

if __name__ == "__main__":
    runpy.run_module("fme.ace.train", run_name="__main__", alter_sys=True)
