"""Differentiable one-dimensional GPS-RO bending-angle forward operator
(PyTorch port of the `long-window-4dvar` (JAX/NeuralGCM) reference repo's
`gpsro_operator.py`). See integrate_gpsro.md for the design and what
changed in the port.

The Abel-layer formulas follow the exponential-refractivity 1D formulation
documented in the ROM SAF ROPP 12 forward-model guide (same reference the
JAX version cites). ROPP itself is not a runtime dependency.

Model columns are represented relative to the local ray-geometry radius
(`earth_radius_curvature_m + geoid_undulation_m`). Keeping refractional and
impact heights in this centered coordinate avoids subtracting two roughly
6.4e6 metre float32 values inside the operator.

**Port note**: the JAX version implements its own `erfcx` (scaled
complementary error function) by hand, in two branches, specifically
because `jax.scipy.special` has no native primitive for it. PyTorch DOES
(`torch.special.erfcx`, a proper numerically stable implementation) --
every call site below uses it directly instead of porting that manual
workaround. The JAX version also flattens every leading batch dimension
and uses `jax.vmap` over individual columns/layers (`_simulate_one_bending_angle`,
`jax.vmap(exponential_layer_bending_angle, ...)`) because that's how JAX
vectorizes a scalar-signature function; PyTorch broadcasts natively, so
this port keeps the layer axis as an ordinary trailing tensor dimension
and reduces over it with `.sum(dim=-1)` -- no flatten/vmap/reshape
machinery needed anywhere.
"""

import torch


GPSRO_OPERATOR_VERSION = 'gpsro-1d-exponential-v1-torch'

# ROPP's default ideal-gas neutral-refractivity constants. With K1 == K3,
# the dry-pressure and e/T terms combine to K1 * total_pressure / T.
REFRACTIVITY_K1_K_PER_HPA = 77.60
REFRACTIVITY_K2_K2_PER_HPA = 3.73e5
REFRACTIVITY_K3_K_PER_HPA = 77.60
WATER_DRY_MOLECULAR_WEIGHT_RATIO = 0.622

STANDARD_GRAVITY_M_S2 = 9.80665
WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_FLATTENING = 0.003352811
WGS84_GRAVITY_RATIO = 0.003449787
WGS84_EQUATORIAL_GRAVITY_M_S2 = 9.7803253359
SOMIGLIANA_CONSTANT = 1.931853e-3
WGS84_ECCENTRICITY = 0.081819

MINIMUM_INVERSE_SCALE_HEIGHT_M1 = 1.e-6
MINIMUM_LAYER_THICKNESS_M = 10.

GPSRO_COLUMN_THERMODYNAMIC_INVALID = 1
GPSRO_COLUMN_REFRACTIVITY_INVALID = 2
GPSRO_COLUMN_REFRACTIONAL_HEIGHT_INVALID = 4
GPSRO_COLUMN_TANGENT_UNSUPPORTED = 8
GPSRO_COLUMN_TOP_GRADIENT_INVALID = 16
GPSRO_COLUMN_SUPER_REFRACTION = 32
GPSRO_COLUMN_VALIDITY_FLAG_MASKS = (
    GPSRO_COLUMN_THERMODYNAMIC_INVALID, GPSRO_COLUMN_REFRACTIVITY_INVALID,
    GPSRO_COLUMN_REFRACTIONAL_HEIGHT_INVALID, GPSRO_COLUMN_TANGENT_UNSUPPORTED,
    GPSRO_COLUMN_TOP_GRADIENT_INVALID, GPSRO_COLUMN_SUPER_REFRACTION,
)
GPSRO_COLUMN_VALIDITY_FLAG_MEANINGS = (
    'thermodynamic_invalid refractivity_invalid '
    'refractional_height_invalid tangent_unsupported '
    'top_refractivity_gradient_invalid super_refraction'
)

# GSI's own critical refractivity gradient (setupbend.f90's `crit_grad`,
# N-units/km) -- the classic literature value at which a ray becomes
# trapped/ducted (super-refraction). GSI flags "close-to-SR" layers at
# 0.75 * this value (setupbend.f90 line ~571); see
# `super_refraction_detected` below and integrate_gpsro.md.
GSI_CRITICAL_REFRACTIVITY_GRADIENT_N_PER_KM = 157.0
GSI_SUPER_REFRACTION_DETECTION_FRACTION = 0.75
# GSI only screens for super-refraction below this impact height
# (setupbend.f90's `alt <= five` gate, km).
GSI_SUPER_REFRACTION_SCREEN_HEIGHT_M = 5000.


def water_vapor_partial_pressure_hpa(specific_humidity, pressure_hpa):
    """Water-vapour partial pressure from specific humidity (kg/kg) and
    total pressure (hPa)."""
    q = specific_humidity
    epsilon = WATER_DRY_MOLECULAR_WEIGHT_RATIO
    return pressure_hpa * q / (epsilon + (1. - epsilon) * q)


def neutral_refractivity(temperature_k, specific_humidity, pressure_hpa):
    """Ideal-gas neutral refractivity in N-units."""
    vapor_pressure = water_vapor_partial_pressure_hpa(specific_humidity, pressure_hpa)
    dry_pressure = pressure_hpa - vapor_pressure
    return (
        REFRACTIVITY_K1_K_PER_HPA * dry_pressure / temperature_k
        + REFRACTIVITY_K2_K2_PER_HPA * vapor_pressure / temperature_k ** 2
        + REFRACTIVITY_K3_K_PER_HPA * vapor_pressure / temperature_k
    )


def surface_gravity_m_s2(latitude_degrees):
    """WGS-84 normal gravity at the ellipsoid surface."""
    sin_squared = torch.sin(torch.deg2rad(latitude_degrees)) ** 2
    return (
        WGS84_EQUATORIAL_GRAVITY_M_S2 * (1. + SOMIGLIANA_CONSTANT * sin_squared)
        / torch.sqrt(1. - WGS84_ECCENTRICITY ** 2 * sin_squared)
    )


def effective_earth_radius_m(latitude_degrees):
    """Latitude-dependent effective radius used for gravity-height."""
    sin_squared = torch.sin(torch.deg2rad(latitude_degrees)) ** 2
    return WGS84_SEMI_MAJOR_AXIS_M / (
        1. + WGS84_FLATTENING + WGS84_GRAVITY_RATIO - 2. * WGS84_FLATTENING * sin_squared
    )


def geopotential_to_geometric_height_m(geopotential_m2_s2, latitude_degrees):
    """Convert absolute geopotential to geometric height above its datum."""
    geopotential_height = geopotential_m2_s2 / STANDARD_GRAVITY_M_S2
    radius = effective_earth_radius_m(latitude_degrees)
    gravity_ratio = surface_gravity_m_s2(latitude_degrees) / STANDARD_GRAVITY_M_S2
    return radius * geopotential_height / (gravity_ratio * radius - geopotential_height)


def refractional_height_m(geometric_height_m, refractivity, radius_base_m):
    """Return ``nr - radius_base`` without forming and subtracting full
    radii (`n = 1 + 1e-6*refractivity`, `r = radius_base + height`)."""
    return geometric_height_m + 1.e-6 * refractivity * (radius_base_m + geometric_height_m)


def _nonnegative_sqrt(value):
    """sqrt(max(value, 0)) with a finite derivative at zero."""
    positive = value > 0.
    safe_value = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, torch.sqrt(safe_value), torch.zeros_like(value))


def exponential_layer_bending_angle(
    impact_height_m, radius_base_m,
    lower_refractional_height_m, upper_refractional_height_m,
    lower_refractivity, upper_refractivity,
):
    """Bending contribution from one exponentially decreasing layer.

    All arguments broadcast together; the layer axis (if any) is just an
    ordinary trailing tensor dimension -- callers reduce over it with
    `.sum(dim=-1)` (see `abel_bending_angle`), no vmap needed.
    """
    x0, x1 = lower_refractional_height_m, upper_refractional_height_m
    n0, n1 = lower_refractivity, upper_refractivity
    thickness = x1 - x0
    safe_thickness = torch.clamp(thickness, min=MINIMUM_LAYER_THICKNESS_M)
    tiny = torch.finfo(impact_height_m.dtype).tiny
    scale_inverse = torch.log(torch.clamp(n0, min=tiny) / torch.clamp(n1, min=tiny)) / safe_thickness
    scale_inverse = torch.clamp(scale_inverse, min=MINIMUM_INVERSE_SCALE_HEIGHT_M1)

    lower = torch.maximum(x0, impact_height_m)
    upper = x1
    lower_separation = torch.clamp(lower - impact_height_m, min=0.)
    upper_separation = torch.clamp(upper - impact_height_m, min=0.)
    z0 = _nonnegative_sqrt(scale_inverse * lower_separation)
    z1 = _nonnegative_sqrt(scale_inverse * upper_separation)

    # Algebraically exp(z0**2) * (erf(z1) - erf(z0)); the erfcx form avoids
    # losing every upper-layer contribution when both error functions round
    # to one in float32.
    scaled_erf_difference = (
        torch.special.erfcx(z0) - torch.exp(z0 ** 2 - z1 ** 2) * torch.special.erfcx(z1)
    )
    refractivity_at_lower = n0 * torch.exp(-scale_inverse * (lower - x0))
    physical_impact = torch.clamp(radius_base_m + impact_height_m, min=1.)
    contribution = (
        1.e-6 * torch.sqrt(2. * torch.pi * physical_impact * scale_inverse)
        * refractivity_at_lower * scaled_erf_difference
    )
    active = (upper > lower) & (thickness >= MINIMUM_LAYER_THICKNESS_M) & (n0 > 0.) & (n1 > 0.)
    return torch.where(active, contribution, torch.zeros_like(contribution))


def increasing_layer_bending_angle(
    impact_height_m, radius_base_m,
    lower_refractional_height_m, upper_refractional_height_m,
    lower_refractivity, upper_refractivity,
):
    """Bending contribution from one constant-gradient refractivity layer
    (used instead of the exponential form where refractivity increases
    with height -- the exponential scale-height formula is ill-posed
    there)."""
    x0, x1 = lower_refractional_height_m, upper_refractional_height_m
    n0, n1 = lower_refractivity, upper_refractivity
    thickness = x1 - x0
    safe_thickness = torch.clamp(thickness, min=MINIMUM_LAYER_THICKNESS_M)
    gradient = (n1 - n0) / safe_thickness
    lower = torch.maximum(x0, impact_height_m)
    upper = x1
    physical_impact = torch.clamp(radius_base_m + impact_height_m, min=1.)
    contribution = (
        -2.e-6 * torch.sqrt(2. * physical_impact) * gradient
        * (_nonnegative_sqrt(upper - impact_height_m) - _nonnegative_sqrt(lower - impact_height_m))
    )
    active = (upper > lower) & (thickness >= MINIMUM_LAYER_THICKNESS_M) & (n0 > 0.) & (n1 >= n0)
    return torch.where(active, contribution, torch.zeros_like(contribution))


def exponential_top_bending_angle(
    impact_height_m, radius_base_m, top_refractional_height_m,
    top_refractivity, inverse_scale_height_m1,
):
    """Bending above the model top under exponential extrapolation."""
    scale_inverse = torch.clamp(inverse_scale_height_m1, min=MINIMUM_INVERSE_SCALE_HEIGHT_M1)
    lower = torch.maximum(top_refractional_height_m, impact_height_m)
    lower_separation = torch.clamp(lower - impact_height_m, min=0.)
    z = _nonnegative_sqrt(scale_inverse * lower_separation)
    refractivity_at_lower = top_refractivity * torch.exp(-scale_inverse * (lower - top_refractional_height_m))
    physical_impact = torch.clamp(radius_base_m + impact_height_m, min=1.)
    contribution = (
        1.e-6 * torch.sqrt(2. * torch.pi * physical_impact * scale_inverse)
        * refractivity_at_lower * torch.special.erfcx(z)
    )
    return torch.where(top_refractivity > 0., contribution, torch.zeros_like(contribution))


def abel_bending_angle(impact_height_m, radius_base_m, refractional_height_profile_m, refractivity_profile):
    """Evaluate 1D Abel bending for a batch of bottom-to-top atmospheric
    columns.

    `refractional_height_profile_m`/`refractivity_profile`:
    (..., n_levels), bottom-to-top (strictly increasing height), at least 2
    levels. `impact_height_m`/`radius_base_m`: (...,), broadcasting against
    the leading batch dims. Column validity is a separate, deliberately
    frozen host-side QC decision (`bending_angle_columns_are_valid`) --
    this function keeps unsupported inputs finite but does not silently
    classify them as assimilable.
    """
    x, n = refractional_height_profile_m, refractivity_profile
    x0, x1 = x[..., :-1], x[..., 1:]
    n0, n1 = n[..., :-1], n[..., 1:]
    impact = impact_height_m[..., None]
    radius = radius_base_m[..., None]

    decreasing = n1 < n0
    exponential = exponential_layer_bending_angle(impact, radius, x0, x1, n0, n1)
    increasing = increasing_layer_bending_angle(impact, radius, x0, x1, n0, n1)
    layer_sum = torch.where(decreasing, exponential, increasing).sum(dim=-1)

    top_thickness = torch.clamp(x[..., -1] - x[..., -2], min=MINIMUM_LAYER_THICKNESS_M)
    tiny = torch.finfo(impact_height_m.dtype).tiny
    top_inverse_scale_height = (
        torch.log(torch.clamp(n[..., -2], min=tiny) / torch.clamp(n[..., -1], min=tiny)) / top_thickness
    )
    top = exponential_top_bending_angle(
        impact_height_m, radius_base_m, x[..., -1], n[..., -1], top_inverse_scale_height,
    )
    return layer_sum + top


def _column_state(temperature_k, specific_humidity, geopotential_m2_s2, pressure_hpa, latitude_degrees, earth_radius_curvature_m, geoid_undulation_m):
    """Return centered ray geometry for a batch of model columns, in
    whatever level order the inputs are given (this repo's backends
    decode bottom-to-top -- surface first, model top last; see
    integrate_gpsro.md) -- shared by `simulate_bending_angles` and the
    validity checks (mirrors the JAX reference's `_one_column_state`,
    which assumed the opposite, top-to-bottom, NeuralGCM convention and
    flipped before calling `abel_bending_angle`; this port's callers pass
    bottom-to-top model fields straight through instead)."""
    refractivity = neutral_refractivity(temperature_k, specific_humidity, pressure_hpa)
    geometric_height = geopotential_to_geometric_height_m(geopotential_m2_s2, latitude_degrees[..., None])
    radius_base = earth_radius_curvature_m + geoid_undulation_m
    refractional_height = refractional_height_m(geometric_height, refractivity, radius_base[..., None])
    return radius_base, refractivity, refractional_height


def bending_angle_column_validity_diagnostics(
    temperature_k, specific_humidity, geopotential_m2_s2, pressure_hpa,
    latitude_degrees, earth_radius_curvature_m, geoid_undulation_m, impact_height_m,
    top_extrapolation_m=0.,
    critical_gradient_n_per_km=GSI_CRITICAL_REFRACTIVITY_GRADIENT_N_PER_KM,
    sr_detection_fraction=GSI_SUPER_REFRACTION_DETECTION_FRACTION,
    sr_screen_height_m=GSI_SUPER_REFRACTION_SCREEN_HEIGHT_M,
):
    """Return per-column failure bits and the first offending layer.

    All model-column arguments: (..., n_levels), bottom-to-top (this
    repo's native pressure-level decode order -- surface first, model top
    last). Geometry/impact height: (...,). `pressure_hpa`: (n_levels,),
    broadcasts against every column, same order.

    `top_extrapolation_m` (default 0, i.e. unchanged behavior): widens
    `tangent_supported` to allow a tangent point up to this many metres
    ABOVE the model's own top refractional height, trusting
    `exponential_top_bending_angle`'s analytic continuation of the top
    layer's own exponential scale height that far -- GSI reaches a FIXED
    55/45 km ceiling the equivalent way, by extending its own discrete
    grid (`nsig_ext`, `setupbend.f90`) rather than rejecting outright (see
    integrate_gpsro.md "GSI comparison"). No operator-math change is
    needed here: `exponential_top_bending_angle` already extrapolates
    correctly for an impact height above `refractional_height[..., -1]`
    (confirmed directly) -- this parameter only widens how far the
    validity mask is willing to trust that extrapolation.

    `critical_gradient_n_per_km`/`sr_detection_fraction`/
    `sr_screen_height_m`: a simplified port of GSI's super-refraction (SR)
    screen (`setupbend.f90`'s `crit_grad`/`qc_layer_SR`, ~line 560-580) --
    reject (`GPSRO_COLUMN_SUPER_REFRACTION`) any ray whose own impact
    height is below `sr_screen_height_m` AND which has, at or below that
    impact height, a layer whose |refractivity gradient| exceeds
    `sr_detection_fraction * critical_gradient_n_per_km` (GSI's own
    critical value, 157 N-units/km, at which a ray becomes trapped/
    ducted). Simplified relative to GSI: reject-only, no relocation of
    the ray above a detected SR layer (GSI's fuller behavior) -- see
    integrate_gpsro.md. Set `critical_gradient_n_per_km` to `None`/`0` to
    disable this check entirely.
    """
    temperature, humidity, geopotential, latitude = (
        temperature_k, specific_humidity, geopotential_m2_s2, latitude_degrees
    )
    radius_base, refractivity, refractional_height = _column_state(
        temperature, humidity, geopotential, pressure_hpa, latitude,
        earth_radius_curvature_m, geoid_undulation_m,
    )

    thermodynamic_valid = (
        torch.isfinite(temperature).all(dim=-1)
        & ((temperature > 100.) & (temperature < 400.)).all(dim=-1)
        & torch.isfinite(humidity).all(dim=-1)
        & ((humidity >= 0.) & (humidity < 1.)).all(dim=-1)
        & torch.isfinite(geopotential).all(dim=-1)
        & torch.isfinite(radius_base) & (radius_base > 5.e6) & (radius_base < 8.e6)
    )
    height_difference = refractional_height[..., 1:] - refractional_height[..., :-1]
    refractivity_valid = (
        torch.isfinite(refractivity).all(dim=-1)
        & (refractivity > 0.).all(dim=-1)
    )
    height_valid = (
        torch.isfinite(refractional_height).all(dim=-1)
        & (height_difference >= MINIMUM_LAYER_THICKNESS_M).all(dim=-1)
    )
    tangent_supported = (
        torch.isfinite(impact_height_m)
        & (impact_height_m >= refractional_height[..., 0])
        & (impact_height_m <= refractional_height[..., -1] + float(top_extrapolation_m))
    )
    top_gradient_valid = refractivity[..., -1] < refractivity[..., -2]

    if critical_gradient_n_per_km:
        safe_height_difference = torch.clamp(height_difference, min=MINIMUM_LAYER_THICKNESS_M)
        gradient_n_per_km = 1000. * (refractivity[..., 1:] - refractivity[..., :-1]) / safe_height_difference
        critical_layer = gradient_n_per_km.abs() >= (float(sr_detection_fraction) * float(critical_gradient_n_per_km))
        layer_at_or_below_impact = refractional_height[..., :-1] <= impact_height_m[..., None]
        super_refraction = (
            (impact_height_m <= float(sr_screen_height_m))
            & (critical_layer & layer_at_or_below_impact).any(dim=-1)
        )
    else:
        super_refraction = torch.zeros_like(impact_height_m, dtype=torch.bool)

    flags = torch.zeros_like(impact_height_m, dtype=torch.int16)
    flags = flags | torch.where(thermodynamic_valid, 0, GPSRO_COLUMN_THERMODYNAMIC_INVALID).to(torch.int16)
    flags = flags | torch.where(refractivity_valid, 0, GPSRO_COLUMN_REFRACTIVITY_INVALID).to(torch.int16)
    flags = flags | torch.where(height_valid, 0, GPSRO_COLUMN_REFRACTIONAL_HEIGHT_INVALID).to(torch.int16)
    flags = flags | torch.where(tangent_supported, 0, GPSRO_COLUMN_TANGENT_UNSUPPORTED).to(torch.int16)
    flags = flags | torch.where(top_gradient_valid, 0, GPSRO_COLUMN_TOP_GRADIENT_INVALID).to(torch.int16)
    flags = flags | torch.where(~super_refraction, 0, GPSRO_COLUMN_SUPER_REFRACTION).to(torch.int16)

    bad_layer = (
        ~torch.isfinite(refractional_height[..., :-1])
        | ~torch.isfinite(refractional_height[..., 1:])
        | (height_difference < MINIMUM_LAYER_THICKNESS_M)
    )
    any_bad = bad_layer.any(dim=-1)
    first_bad_layer = torch.where(
        any_bad, bad_layer.to(torch.int16).argmax(dim=-1).to(torch.int16),
        torch.full_like(any_bad, -1, dtype=torch.int16),
    )
    return flags, first_bad_layer


def bending_angle_columns_are_valid(
    temperature_k, specific_humidity, geopotential_m2_s2, pressure_hpa,
    latitude_degrees, earth_radius_curvature_m, geoid_undulation_m, impact_height_m,
    **validity_kwargs,
):
    flags, _ = bending_angle_column_validity_diagnostics(
        temperature_k, specific_humidity, geopotential_m2_s2, pressure_hpa,
        latitude_degrees, earth_radius_curvature_m, geoid_undulation_m, impact_height_m,
        **validity_kwargs,
    )
    return flags == 0


def simulate_bending_angles(
    temperature_k, specific_humidity, geopotential_m2_s2, pressure_hpa,
    latitude_degrees, earth_radius_curvature_m, geoid_undulation_m, impact_height_m, mask,
):
    """Simulate a batch of bending angles.

    Model variables: (..., n_levels), bottom-to-top pressure-level order
    (this repo's native `decoded[base]` order). Geometry, impact height,
    and mask: (...,). A distinct model column may be supplied for every
    retained occultation level (the caller interpolates one per (profile,
    level) at that level's own, possibly drifting, tangent-point location
    -- see get_gpsro).

    Masked-out rays are substituted with a deliberately valid,
    state-independent standard column/ray BEFORE evaluation (not just
    masked in the output): the Abel integral's divisions/sqrts/logs can
    produce a finite forward value but a non-finite local gradient for
    genuinely invalid geometry (e.g. refractional heights that reverse
    near the model top), and masking only the result is too late --
    reverse-mode autodiff can still encounter `0 * NaN` and contaminate
    the gradient of the whole loss. Ported directly from the JAX
    reference's `simulate_bending_angles`, which hit exactly this failure
    mode.
    """
    dtype = temperature_k.dtype
    device = temperature_k.device
    n_levels = temperature_k.shape[-1]
    mask = mask.to(torch.bool)
    column_mask = mask[..., None]

    standard_temperature = torch.full((n_levels,), 250., dtype=dtype, device=device)
    standard_humidity = torch.zeros((n_levels,), dtype=dtype, device=device)
    standard_latitude = torch.zeros((), dtype=dtype, device=device)
    effective_radius = effective_earth_radius_m(standard_latitude)
    gravity_ratio = surface_gravity_m_s2(standard_latitude) / STANDARD_GRAVITY_M_S2
    standard_height = torch.linspace(0., 50000., n_levels, dtype=dtype, device=device)
    standard_geopotential_height = (
        standard_height * gravity_ratio * effective_radius / (effective_radius + standard_height)
    )
    standard_geopotential = standard_geopotential_height * STANDARD_GRAVITY_M_S2

    safe_temperature = torch.where(column_mask, temperature_k, standard_temperature)
    safe_humidity = torch.where(column_mask, specific_humidity, standard_humidity)
    safe_geopotential = torch.where(column_mask, geopotential_m2_s2, standard_geopotential)
    safe_latitude = torch.where(mask, latitude_degrees, standard_latitude)
    safe_curvature = torch.where(mask, earth_radius_curvature_m, torch.full_like(earth_radius_curvature_m, 6371000.))
    safe_undulation = torch.where(mask, geoid_undulation_m, torch.zeros_like(geoid_undulation_m))
    safe_impact_height = torch.where(mask, impact_height_m, torch.full_like(impact_height_m, 10000.))

    radius_base, refractivity, refractional_height = _column_state(
        safe_temperature, safe_humidity, safe_geopotential, pressure_hpa, safe_latitude,
        safe_curvature, safe_undulation,
    )
    values = abel_bending_angle(safe_impact_height, radius_base, refractional_height, refractivity)
    return torch.where(mask, values, torch.zeros_like(values))
