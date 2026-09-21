# lw4dvar-torch

Prototype long-window 4dvar solver (no background or model error terms in loss).

Compute optimal initial conditions for torch-based AI forecast models using an ob-term only loss.
Options include AIFS-single-1.1, AIFS-single-2.0, FourCastNet3 and Microsoft Aurora
(via the `model_backend={aifs,fcn3,aurora}` yaml config parameter).

AIFS-single-2.0 is validated, FourCastNet3 and Aurora are works in progress.

Currently only surface pressure observations are assimilated.
