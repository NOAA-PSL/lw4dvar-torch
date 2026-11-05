"""Two-head microwave observation operator (ATMS, AMSU-A), pure torch.

From radianced `atmsdec/operator.py` (NOAA-PSL/radianced); it must import
nothing but torch. Extended here (2026-10-18) for the other microwave
sensors: the channel count comes from the bundle (`nch`, default 22 = ATMS),
a decoder with satellite one-hot inputs (`sat-0`, `sat-1`, ...) gets them from
`obs['sat_id']`, and log sigma_k is clamped to radianced's LOGS_RANGE.
Bundles "atms_twohead/1" and "twohead/2" (`export_mw_bundle.py`) both load.

Model (radianced `scripts/twohead.py`, `--vert none`):

    x (physical units) -> normalize                    x_n = (x - x_mean) / x_std
    surface head   P(x_n)        -> e_hat (nch emissivities), log sigma_k (k = 1..K)
    emissivity     e = e_hat + U z                     z: (n_obs, K) control, prior N(0, sigma_k^2)
    radiance head  D(x_n, (e - e_mu) / e_sd) -> Tb_n;  Tb = Tb_n * y_std + y_mean   (K, target y_o - BC)

Both heads are Linear + BatchNorm1d + SiLU MLPs, BatchNorm always in eval mode.

Use:
    op = AtmsOperator.load("atms_twohead_bo_s1.pt", device)
    x = op.assemble(fields, obs)          # (n_obs, n_inputs), physical units
    tb, e_hat, log_sig = op(x, z=None)    # (n_obs, nch) K, (n_obs, nch), (n_obs, K)

`fields` (model side, per obs, already interpolated to the obs location and time):
    't'  (n_t, n_obs) K on op.t_levels_hpa (top -> bottom, 10 ... 1000 hPa)
    'q'  (n_q, n_obs) kg/kg on op.q_levels_hpa (50 ... 1000 hPa), or on op.q_levels_hpa[1:]
         (100 ... 1000 hPa, the AIFS 1-degree prognostic set): the missing top level is then
         filled with its training mean
    'sp' Pa, 'msl' Pa, 'skt' '2t' '2d' K, '10u' '10v' m/s, 'tcw' kg/m2,
    'lsm', 'z' (surface geopotential, m2/s2), 'sdor', 'slor'       each (n_obs,)
`obs` (obs file side, each (n_obs,)):
    'land_frac' 'water_frac' 'ice_frac' 'snow_frac', 'Sat_Zenith_Angle' 'Scan_Angle'
    'Sat_Azimuth_Angle' (degrees); optional 'Obs_Time' (hours, default 0: the state is at the obs
    time) and 'sat_id' (ATMS: 0 = S-NPP, 1 = NOAA-20; other sensors: index into the decoder's
    satellite list, one-hot encoded; default 0).
Gradients flow from tb to every model field except through the above-ground masks (a step
function of sp).
"""

import torch

BUNDLE_VERSION = "twohead/2"
BUNDLE_VERSIONS = ("atms_twohead/1", BUNDLE_VERSION)
NCH = 22  # ATMS; a bundle's own `nch` overrides
LOGS_RANGE = (-12.0, 5.0)  # radianced scripts/twohead.py

# decoder input name -> key in `fields` (model) or `obs`
_MODEL_SURFACE = {"skt": "skt", "t2m": "2t", "d2m": "2d", "sp": "sp", "msl": "msl", "u10": "10u",
                  "v10": "10v", "tcw": "tcw", "lsm": "lsm", "zsfc": "z", "sdor": "sdor", "slor": "slor",
                  # the 37-level A1/A2 decoders also take these (same names in the model)
                  **{k: k for k in ("sd", "siconc", "swvl1", "stl1", "cvh", "cvl", "lai_hv", "lai_lv", "slt", "fsr")}}
_OBS = ("land_frac", "water_frac", "ice_frac", "snow_frac", "Sat_Zenith_Angle", "Scan_Angle",
        "Sat_Azimuth_Angle")
_OBS_DEFAULT = {"Obs_Time": 0.0, "sat_id": 0.0}


class _MLP(torch.nn.Module):
    """Same module list as deep-obs H_MLP with SiLU, so its state dict loads unchanged."""

    def __init__(self, n_in, n_out, hidden):
        super().__init__()
        layers = []
        for h in hidden:
            layers += [torch.nn.Linear(n_in, h), torch.nn.BatchNorm1d(h), torch.nn.SiLU()]
            n_in = h
        layers.append(torch.nn.Linear(n_in, n_out))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AtmsOperator(torch.nn.Module):
    def __init__(self, bundle):
        super().__init__()
        if bundle.get("version") not in BUNDLE_VERSIONS:
            raise ValueError(f"bundle version {bundle.get('version')!r}, this operator file reads {BUNDLE_VERSIONS}")
        self.sensor = str(bundle.get("sensor", "atms"))
        self.nch = int(bundle.get("nch", NCH))
        self.input_features = list(bundle["input_features"])
        self.n_sat = sum(f.startswith("sat-") for f in self.input_features)
        self.t_levels_hpa = [float(p) for p in bundle["t_levels_hpa"]]
        self.q_levels_hpa = [float(p) for p in bundle["q_levels_hpa"]]
        # level k's feature suffix (tmp-<i>, above-<i>, spfh-<i>), top -> bottom: i = k for Bo's 14 levels and A1's
        # 37, an index into the 37 ERA5 levels when a decoder takes only some of them (A2: i = 5..36)
        self._t_ids = sorted(int(f[4:]) for f in self.input_features if f.startswith("tmp-"))
        self._q_ids = sorted(int(f[5:]) for f in self.input_features if f.startswith("spfh-"))
        self.channel_sets = {k: list(v) for k, v in bundle["channel_sets"].items()}  # 1-based channels
        # decoder surface input -> `fields` key, for the inputs this bundle takes
        self.surface_inputs = {n: k for n, k in _MODEL_SURFACE.items() if n in self.input_features}
        self.K = int(bundle["K"])
        n_in = len(self.input_features)
        self.P = _MLP(n_in, self.nch + self.K, bundle["hidden"])
        self.D = _MLP(n_in + self.nch, self.nch, bundle["hidden"])
        self.P.load_state_dict(bundle["P"])
        self.D.load_state_dict(bundle["D"])
        for name in ("x_mean", "x_std", "y_mean", "y_std", "e_mu", "e_sd", "U", "sigma_o"):
            self.register_buffer(name, torch.as_tensor(bundle[name], dtype=torch.float32).clone())
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def load(cls, path, device="cpu"):
        return cls(torch.load(path, map_location="cpu", weights_only=True)).to(device)

    def train(self, mode=True):
        # BatchNorm must stay in eval mode (running statistics), whatever the caller does
        return super().train(False)

    def assemble(self, fields, obs):
        """Stack the decoder inputs, in `input_features` order, physical units -> (n_obs, n_in)."""
        t, q, sp = fields["t"], fields["q"], fields["sp"]
        if t.shape[0] != len(self.t_levels_hpa):
            raise ValueError(f"t has {t.shape[0]} levels, expected {len(self.t_levels_hpa)} ({self.t_levels_hpa})")
        n_obs = sp.shape[0]
        n_top = len(self.q_levels_hpa) - q.shape[0]
        if 0 < n_top <= len(self.q_levels_hpa):
            # levels the model doesn't carry (above its top q level) at the decoder's mean
            top = torch.stack([self.x_mean[self.input_features.index("spfh-%d" % self._q_ids[k])]
                               for k in range(n_top)])
            q = torch.cat([top.to(q.dtype)[:, None].expand(n_top, n_obs), q], 0)
        elif n_top != 0:
            raise ValueError(f"q has {q.shape[0]} levels, expected {len(self.q_levels_hpa)} ({self.q_levels_hpa})")
        cols = {}
        for k in range(len(self.t_levels_hpa)):
            cols[f"tmp-{self._t_ids[k]}"] = t[k]
            cols[f"above-{self._t_ids[k]}"] = (self.t_levels_hpa[k] < sp.detach() / 100.0).to(t.dtype)
        for k in range(len(self.q_levels_hpa)):
            cols[f"spfh-{self._q_ids[k]}"] = q[k]
        for name, key in self.surface_inputs.items():
            cols[name] = fields[key]
        for name in _OBS:
            cols[name] = torch.as_tensor(obs[name], dtype=t.dtype, device=t.device)
        for name, default in _OBS_DEFAULT.items():
            v = obs.get(name, default)
            cols[name] = torch.as_tensor(v, dtype=t.dtype, device=t.device).expand(n_obs)
        for k in range(self.n_sat):
            cols[f"sat-{k}"] = (cols["sat_id"] == k).to(t.dtype)
        return torch.stack([cols[f] for f in self.input_features], 1)

    def forward(self, x, z=None):
        """x: (n_obs, n_in) physical units; z: (n_obs, K) or None (= 0).
        Returns Tb (n_obs, nch) in K, e_hat (n_obs, nch), log sigma_k (n_obs, K)."""
        xn = (x - self.x_mean) / self.x_std
        o = self.P(xn)
        e_hat = o[:, :self.nch] * self.e_sd + self.e_mu
        log_sig = o[:, self.nch:].clamp(*LOGS_RANGE)
        e = e_hat if z is None else e_hat + z @ self.U.T
        tb = self.D(torch.cat([xn, (e - self.e_mu) / self.e_sd], 1)) * self.y_std + self.y_mean
        return tb, e_hat, log_sig
