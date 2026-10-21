"""GSI convinfo reader: per observation type use flag and gross-error limits.

The dated convinfo files come from NOAA-PSL/build_gsinfo
(https://github.com/NOAA-PSL/build_gsinfo, `convinfo/historical/
global_convinfo.txt.<YYYYMMDDHH>`, each valid from its date until the next
one). Rows are keyed by (otype, PREPBUFR report type, subtype), e.g.
('t', 181, 0); the columns used here are

  iuse   1 use, -1 monitor, 0 do not use, -3 skip reading
  gross  gross-error ratio
  ermax, ermin  bounds on the observation error used in the gross check

GSI's gross check rejects an observation when
|O - B| / clip(error, ermin, ermax) > gross (see `gross_check`).
"""

import glob
import os

import numpy as np


DEFAULT_GSINFO = '/scratch4/BMC/gsienkf/Sergey.Frolov/ext/build_gsinfo'


def find_convinfo(gsinfo_root, date):
    """The historical convinfo valid at `date` ('YYYY-MM-DDTHH' or YYYYMMDDHH):
    the file with the latest start date not after it."""
    stamp = str(date).replace('-', '').replace('T', '')[:10]
    files = glob.glob(os.path.join(gsinfo_root, 'convinfo', 'historical', 'global_convinfo.txt.??????????'))
    dated = sorted((os.path.basename(f).rsplit('.', 1)[1], f) for f in files)
    valid = [f for d, f in dated if d <= stamp]
    if not valid:
        raise FileNotFoundError(f'no convinfo in {gsinfo_root}/convinfo/historical valid at {date}')
    return valid[-1]


def read_convinfo(path):
    """{(otype, type, sub): {'iuse', 'twindow', 'gross', 'ermax', 'ermin'}}"""
    table = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 11 or line.lstrip().startswith('!'):
                continue
            try:
                key = (parts[0], int(parts[1]), int(parts[2]))
                row = {'iuse': int(parts[3]), 'twindow': float(parts[4]), 'gross': float(parts[8]),
                       'ermax': float(parts[9]), 'ermin': float(parts[10])}
            except ValueError:
                continue
            table.setdefault(key, row)  # first occurrence wins, as in GSI
    return table


def lookup(table, otype, report_type, subtype=0):
    """The row for (otype, report_type, subtype), falling back to subtype 0;
    None when the type is not in the table."""
    return table.get((otype, int(report_type), int(subtype)), table.get((otype, int(report_type), 0)))


def gross_check(departure, error, row):
    """True where |departure| / clip(error, ermin, ermax) <= gross (passes).
    `row` from `lookup`; a missing row passes everything."""
    if row is None:
        return np.ones(np.shape(departure), bool)
    err = np.clip(error, row['ermin'], row['ermax'])
    return np.abs(departure) <= row['gross'] * err
