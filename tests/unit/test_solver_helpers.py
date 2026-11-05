"""Small solver helpers: obs-density Jo weights, the control-variable mask,
and the date utilities."""

import math
from datetime import timezone

import pytest
import torch

import long_window_4dvar_utils as utils


def density_weights(lats, lons, used, box_deg=10.0):
    f = lambda v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    return utils._obs_density_weights(torch.tensor(used), f(lats), f(lons), box_deg)


def test_density_weights_share_a_box():
    # two obs in one 10-deg box, one alone in another at the same latitude,
    # one QC-rejected, one padding slot (lat/lon = 1e10)
    w = density_weights([5.0, 6.0, 5.0, 5.0, 1e10], [10.0, 12.0, 100.0, 200.0, 1e10],
                        [True, True, True, False, False])
    torch.testing.assert_close(w, torch.tensor([0.75, 0.75, 1.5, 0.0, 0.0]))


def test_density_weights_area_factor():
    # isolated obs: weights follow the cosine of their box-centre latitude
    w = density_weights([5.0, 75.0], [0.0, 0.0], [True, True])
    assert float(w.sum()) == pytest.approx(2.0, rel=1e-6)
    ratio = math.cos(math.radians(75.0)) / math.cos(math.radians(5.0))
    assert float(w[1] / w[0]) == pytest.approx(ratio, rel=1e-5)


def test_density_weights_wrap_longitude():
    # -170 and 190 deg are the same place
    w = density_weights([5.0, 5.0], [-170.0, 190.0], [True, True])
    torch.testing.assert_close(w, torch.tensor([1.0, 1.0]))


def test_control_mask_channels_first():
    state = torch.ones(1, 3, 2, 2)
    ref = torch.zeros(1, 3, 2, 2)
    keep = torch.tensor([True, False, True])
    out = utils._apply_control_mask(state, keep, ref, 'channels_first')
    assert out[0, :, 0, 0].tolist() == [1.0, 0.0, 1.0]


def test_control_mask_channels_last():
    state = torch.ones(1, 1, 4, 3)
    ref = torch.zeros(1, 1, 4, 3)
    keep = torch.tensor([False, True, True])
    out = utils._apply_control_mask(state, keep, ref, 'channels_last')
    assert out[0, 0, 0].tolist() == [0.0, 1.0, 1.0]


def test_control_mask_unknown_layout():
    with pytest.raises(ValueError, match="state_layout"):
        utils._apply_control_mask(torch.ones(1), torch.tensor([True]), torch.zeros(1), 'nhwc')


@pytest.mark.parametrize("date, hours, expected", [
    ('2015-01-01T00', 6, '2015-01-01T06'),
    ('2015-12-31T18', 12, '2016-01-01T06'),
    ('2016-03-01T00', -24, '2016-02-29T00'),
])
def test_add_hours(date, hours, expected):
    assert utils.add_hours(date, hours) == expected


def test_get_yyyymmddhh():
    assert utils.get_YYYYMMDDHH('2015-01-02T18') == '2015010218'


@pytest.mark.parametrize("backend, aware", [('aifs', False), ('ace2', False), ('fcn3', True), ('sfno', True)])
def test_to_utc_datetime(backend, aware):
    dt = utils._to_utc_datetime('2015-01-02T18', backend)
    assert (dt.tzinfo is timezone.utc) == aware
    assert (dt.year, dt.month, dt.day, dt.hour) == (2015, 1, 2, 18)
