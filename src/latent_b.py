"""Spectral latent-space background-error covariance: diagonal in total
wavenumber l (homogeneous, isotropic), full covariance across latent channels
within wavenumber bands -- the latent analogue of the operational
"horizontal spectral x vertical covariance per wavenumber" B.

    dz = SHT^-1( c ),   c[:, l, m] = a_l * S_b(l) v[:, l, m]

S_b = Sigma_b^{1/2} (symmetric square root) of the per-coefficient channel
covariance pooled over band b; a_l^2 = tr K_l / tr Sigma_b rescales each l to
its own total variance (K_l the per-l covariance). The control v is real,
shape (2, C, L, M) (real / imaginary parts); entries with m > l and the
imaginary part at m = 0 are inert (masked), and m > 0 parts are scaled by
1/sqrt(2), so v ~ N(0, I) gives draws with E[c c^H] matching the estimate
and Jb = ||v||^2.

Built from the `cross` (L, C, C) output of diagnostics/estimate_latent_b.py
(`--cross`). `channel_cov=False` keeps only the diagonal (the spectral-
diagonal B, for comparison).
"""

import numpy as np
import torch
import torch_harmonics as th

DEFAULT_BANDS = (0, 6, 10, 15, 20, 30, 40, 60, 90, 130)


class SpectralChannelB(torch.nn.Module):
    def __init__(self, cross: torch.Tensor, H: int, W: int, grid: str, bands=DEFAULT_BANDS,
                 shrink: float = 0.05, channel_cov: bool = True):
        super().__init__()
        cross = cross.double()
        L, C, _ = cross.shape
        edges = list(bands) + [L]
        self.bands = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        w = 2.0 * torch.arange(L, dtype=torch.float64) + 1.0
        sqrt_mats, amp, self.band_eigs = [], torch.zeros(L, dtype=torch.float64), []
        for lo, hi in self.bands:
            sig = (cross[lo:hi] * w[lo:hi, None, None]).sum(0) / w[lo:hi].sum()
            d = torch.diagonal(sig)
            sig = (1.0 - shrink) * sig + shrink * torch.diag(d) if channel_cov else torch.diag(d)
            e, U = torch.linalg.eigh(sig)
            e = e.clamp(min=0)
            self.band_eigs.append(e.flip(0))
            sqrt_mats.append((U * e.sqrt()) @ U.T)
            amp[lo:hi] = (torch.diagonal(cross[lo:hi], dim1=1, dim2=2).sum(-1) / d.sum()).clamp(min=0).sqrt()
        self.register_buffer('sqrt_mats', torch.stack(sqrt_mats))      # (nband, C, C)
        self.register_buffer('amp', amp)                                 # (L,)
        band_of = torch.zeros(L, dtype=torch.long)
        for i, (lo, hi) in enumerate(self.bands):
            band_of[lo:hi] = i
        self.register_buffer('band_of', band_of)
        self.isht = th.InverseRealSHT(H, W, lmax=L, mmax=W // 2 + 1, grid=grid, norm='ortho').double()
        M = self.isht.mmax
        mask = (torch.arange(M)[None, :] <= torch.arange(L)[:, None]).double()   # (L, M)
        scale = torch.full((M,), 0.5 ** 0.5, dtype=torch.float64)
        scale[0] = 1.0
        self.register_buffer('re_w', mask * scale)
        im = mask * scale
        im[:, 0] = 0.0
        self.register_buffer('im_w', im)
        self.C, self.L, self.M = C, L, M

    @property
    def control_shape(self):
        return (2, self.C, self.L, self.M)

    def coeffs(self, v: torch.Tensor) -> torch.Tensor:
        """Control v (2, C, L, M) -> SHT coefficients (C, L, M) complex."""
        c = torch.complex(v[0] * self.re_w, v[1] * self.im_w)
        out = torch.empty_like(c)
        for i, (lo, hi) in enumerate(self.bands):
            S = self.sqrt_mats[i].to(c.real.dtype)
            out[:, lo:hi] = torch.einsum('cd,dlm->clm', S.to(c.dtype), c[:, lo:hi])
        return out * self.amp.to(c.real.dtype)[None, :, None]

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Control v -> latent increment (C, H, W)."""
        return self.isht(self.coeffs(v))


def localization_sqrt_spectrum(loc_km: float, H: int, W: int, grid: str, lmax: int = None,
                               radius_km: float = 6371.0, tol: float = 1e-6) -> torch.Tensor:
    """sqrt of the Legendre spectrum lambda_l of the isotropic Gaussian
    correlation rho(d) = exp(-d^2 / (2 loc^2)) (d great-circle distance), so
    that F = SHT^-1( sqrt(lambda_l) v ), v white (same convention as
    SpectralChannelB), has unit variance and correlation rho. lambda_l is
    obtained by Funk-Hecke from the orthonormal SHT of rho centred on the
    pole; truncated where it falls below `tol` of its maximum (returned
    length = the l cutoff), then renormalised to unit point variance."""
    sht = th.RealSHT(H, W, lmax=lmax, grid=grid, norm='ortho').double()
    lat = torch.as_tensor(np.degrees(np.arcsin(_grid_sinlat(H, grid))), dtype=torch.float64)
    d = torch.deg2rad(90.0 - lat) * radius_km                       # distance from the north pole
    rho = torch.exp(-0.5 * (d / loc_km) ** 2)[:, None].expand(H, W).contiguous()
    a_l0 = sht(rho[None])[0, :, 0].real                              # m = 0 coefficients
    L = a_l0.shape[0]
    l = torch.arange(L, dtype=torch.float64)
    lam = (torch.sqrt(4 * np.pi / (2 * l + 1)) * a_l0).clamp(min=0)
    keep = int((lam > tol * lam.max()).nonzero().max()) + 1
    lam = lam[:keep]
    lam = lam / ((2 * l[:keep] + 1) * lam).sum() * 4 * np.pi       # unit variance at every point
    return lam.sqrt()


def _grid_sinlat(H: int, grid: str) -> np.ndarray:
    """sin(latitude), north to south, of torch_harmonics' grid (as its SHTs use)."""
    if grid == 'legendre-gauss':
        x, _ = np.polynomial.legendre.leggauss(H)
        return np.sort(x)[::-1].copy()
    if grid == 'equiangular':
        return np.cos(np.linspace(0, np.pi, H))
    raise ValueError(grid)


class EnsembleB(torch.nn.Module):
    """Static-ensemble latent B with horizontal localization:

        B = (1/(N-1)) sum_k (e_k e_k^T) o L,
        dz = (N-1)^-1/2 sum_k e_k * F_k,   F_k = SHT^-1( sqrt(lambda) v_k ),

    e_k the stored latent error samples (mean removed), L the Gaussian
    localization of scale `loc_km` (same for all channels, so the samples'
    own cross-channel and anisotropic -- e.g. geostrophic -- structure is
    kept), v (2, N, Lc, Mc) the white control (Jb = ||v||^2, masking and
    m>0 scaling as SpectralChannelB). Lc is the localization's spectral
    cutoff, so the control is small for broad localization."""

    def __init__(self, samples: torch.Tensor, loc_km: float, grid: str, chunk: int = 32):
        super().__init__()
        N, C, H, W = samples.shape
        self.register_buffer('samples', samples)                     # (N, C, H, W), fp16 ok
        sq = localization_sqrt_spectrum(loc_km, H, W, grid)
        Lc = sq.shape[0]
        self.register_buffer('sqrt_lam', sq)
        self.isht = th.InverseRealSHT(H, W, lmax=Lc, mmax=Lc, grid=grid, norm='ortho').double()
        mask = (torch.arange(Lc)[None, :] <= torch.arange(Lc)[:, None]).double()
        scale = torch.full((Lc,), 0.5 ** 0.5, dtype=torch.float64)
        scale[0] = 1.0
        im = mask * scale
        im[:, 0] = 0.0
        self.register_buffer('re_w', mask * scale)
        self.register_buffer('im_w', im)
        self.N, self.C, self.H, self.W, self.Lc, self.chunk = N, C, H, W, Lc, chunk
        self.loc_km = loc_km

    @property
    def control_shape(self):
        return (2, self.N, self.Lc, self.Lc)

    def weights(self, v: torch.Tensor) -> torch.Tensor:
        """Control -> localized weight fields F (N, H, W)."""
        c = torch.complex(v[0] * self.re_w, v[1] * self.im_w) * self.sqrt_lam.to(v.dtype)[None, :, None]
        return self.isht(c)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        F = (self.weights(v) / (self.N - 1) ** 0.5).float()
        return _WeightedSampleSum.apply(F, self.samples, self.chunk)


class _WeightedSampleSum(torch.autograd.Function):
    """dz[c] = sum_n e[n, c] * F[n] (pointwise), chunked over channels in fp32;
    backward grad_F[n] = sum_c e[n, c] * g[c]. Only the (fp16) samples are
    referenced -- autograd through einsum would save fp32 copies of them."""

    @staticmethod
    def forward(ctx, F, samples, chunk):
        ctx.save_for_backward(samples)
        ctx.chunk = chunk
        C = samples.shape[1]
        return torch.cat([torch.einsum('nchw,nhw->chw', samples[:, c0:c0 + chunk].float(), F)
                          for c0 in range(0, C, chunk)])

    @staticmethod
    def backward(ctx, g):
        (samples,) = ctx.saved_tensors
        chunk = ctx.chunk
        gF = torch.zeros((samples.shape[0],) + tuple(samples.shape[2:]), dtype=g.dtype, device=g.device)
        for c0 in range(0, samples.shape[1], chunk):
            gF += torch.einsum('nchw,chw->nhw', samples[:, c0:c0 + chunk].float(), g[c0:c0 + chunk])
        return gF, None, None
