"""Adjoint dot-product tests: <A v, w> = <v, A^T w> for the toy model's
tangent-linear / adjoint (the gate a backend with a hand-written backward,
e.g. a custom VJP, must pass) and for the custom autograd Function in
latent_b.py. Each has a planted bug it must catch."""

import torch

import latent_b
from gates import dot_product_error, model_tangent_and_adjoint
from toy_model import ToyModel

DOT_TOL = 1e-12


def _tangent_and_adjoint(model, steps=3):
    x0 = 0.3 * torch.randn(model.latent_shape, generator=torch.Generator().manual_seed(3),
                           dtype=torch.float64)
    return model_tangent_and_adjoint(model, model.initial_state(), x0, steps)


def test_model_adjoint_dot_product():
    model = ToyModel()
    fwd, adj, y_shape = _tangent_and_adjoint(model)
    assert dot_product_error(fwd, adj, model.latent_shape, y_shape) < DOT_TOL


class _RollWithWrongAdjoint(torch.autograd.Function):
    """roll(+1) east, whose backward rolls east again instead of back west."""

    @staticmethod
    def forward(h):
        return torch.roll(h, 1, dims=-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, g):
        return torch.roll(g, 1, dims=-1)

    @staticmethod
    def jvp(ctx, v):
        return torch.roll(v, 1, dims=-1)


class _ToyWithWrongAdjoint(ToyModel):
    def _step(self, x, increment):
        h = torch.tanh(torch.einsum('lc,bchw->blhw', self.w_enc, (x - self.mean) / self.spread))
        if increment is not None:
            h = h + increment[None]
        h = _RollWithWrongAdjoint.apply(h)
        h = 0.5 * h + 0.25 * (torch.roll(h, 1, dims=-2) + torch.roll(h, -1, dims=-2))
        dx = torch.einsum('cl,blhw->bchw', self.w_dec, torch.tanh(h))
        return x + self.spread * dx


def test_planted_bug_wrong_adjoint_of_advection():
    model = _ToyWithWrongAdjoint()
    fwd, adj, y_shape = _tangent_and_adjoint(model)
    assert dot_product_error(fwd, adj, model.latent_shape, y_shape) > 1e-3


# --- latent_b._WeightedSampleSum (EnsembleB's custom backward) ---------------

N, C, H, W = 5, 7, 6, 8


def _samples():
    return torch.randn(N, C, H, W, generator=torch.Generator().manual_seed(11)).half()


def test_weighted_sample_sum_adjoint():
    samples = _samples()
    weighted_sum = latent_b._WeightedSampleSum

    def forward(F):   # linear in the weights F (N, H, W)
        return weighted_sum.apply(F.float(), samples, 3)

    def adjoint(g):
        F = torch.zeros(N, H, W, requires_grad=True)
        (gF,) = torch.autograd.grad(weighted_sum.apply(F, samples, 3), F, grad_outputs=g.float())
        return gF

    # fp32 arithmetic inside, so rounding-level agreement is ~1e-7
    assert dot_product_error(forward, adjoint, (N, H, W), (C, H, W)) < 1e-5


class _WeightedSampleSumWrongSign(latent_b._WeightedSampleSum):
    @staticmethod
    def backward(ctx, g):
        gF, _, _ = latent_b._WeightedSampleSum.backward(ctx, g)
        return -gF, None, None


def test_planted_bug_weighted_sample_sum_sign():
    samples = _samples()

    def forward(F):
        return _WeightedSampleSumWrongSign.apply(F.float(), samples, 3)

    def adjoint(g):
        F = torch.zeros(N, H, W, requires_grad=True)
        (gF,) = torch.autograd.grad(_WeightedSampleSumWrongSign.apply(F, samples, 3), F,
                                    grad_outputs=g.float())
        return gF

    assert dot_product_error(forward, adjoint, (N, H, W), (C, H, W)) > 1.0
