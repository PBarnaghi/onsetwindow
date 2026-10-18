import math
from pathlib import Path

import numpy as np
import pytest

from onsetwindow import Intervention, ParameterError, load_parameters, simulate, window_of_opportunity
from onsetwindow.cli import main
from onsetwindow.model import kernel_integral, risk_by_age

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "parameters.param"

MINIMAL = """
[simulation]
age_max = 100
dt = 0.05
report_ages = 50, 70

[individual]
sex = {sex}
environment = {env}

[genotype]
G1 = {z}

[beta]
G1 = {beta}

[threshold]
theta0 = {theta0}
a = 2.0

[domain:d1]
weight_F = {wF}
weight_M = {wM}
tau = {tau}
lambda0 = 1.0
lambda_genotype = G1:{lam}
opportunity_ages = 0, 100
opportunity_values = 1, 1
env_baseline = {envb}
env_coef = {envc}
onset_age = {onset}
"""


def write(tmp_path, **kw):
    defaults = dict(sex="F", env=0.0, z=0, beta=0.0, theta0=50.0, wF=1.0, wM=1.0,
                    tau="inf", lam=0.0, envb=1.0, envc=0.0, onset=0)
    defaults.update(kw)
    f = tmp_path / "p.param"
    f.write_text(MINIMAL.format(**defaults))
    return load_parameters(f)


# --------------------------------------------------------------------------- #
# Kernel integral against closed forms
# --------------------------------------------------------------------------- #

def test_kernel_permanent_is_linear():
    dt = 0.1
    x = np.ones(1001)
    I = kernel_integral(x, math.inf, dt)
    t = dt * np.arange(1001)
    assert np.allclose(I, t)


def test_kernel_exponential_closed_form():
    dt, tau = 0.1, 12.0
    x = np.full(1001, 2.0)
    I = kernel_integral(x, tau, dt)
    t = dt * np.arange(1001)
    assert np.allclose(I, 2.0 * tau * (1 - np.exp(-t / tau)), atol=1e-9)


# --------------------------------------------------------------------------- #
# Burden, threshold crossing, rho
# --------------------------------------------------------------------------- #

def test_burden_permanent_constant_exposure(tmp_path):
    p = write(tmp_path)                      # x = 1 from birth, tau = inf, w = 1 -> S = t
    tr = simulate(p)
    assert np.allclose(tr.S, tr.ages)
    assert tr.T == pytest.approx(50.0, abs=1e-6)
    assert tr.at(tr.pi, tr.T) == pytest.approx(0.5, abs=1e-3)


def test_onset_switch(tmp_path):
    p = write(tmp_path, onset=20)            # exposure only from age 20 -> S = t - 20
    tr = simulate(p)
    assert tr.T == pytest.approx(70.0, abs=1e-6)


def test_rho_is_genetic_share_of_threshold(tmp_path):
    p = write(tmp_path, z=1, beta=0.5)       # S = t + 0.5 t = 1.5 t -> T = 50/1.5
    tr = simulate(p)
    assert tr.T == pytest.approx(50 / 1.5, abs=1e-6)
    assert tr.rho == pytest.approx(tr.T * 0.5 / 50.0)
    assert tr.rho == pytest.approx(1 / 3, abs=1e-6)


def test_pibar_is_running_max_and_absorbing(tmp_path):
    # protective-only domain makes S fall; pibar must never decrease
    p = write(tmp_path, wF=-1.0, beta=2.0, z=1, theta0=20.0, tau=5)
    tr = simulate(p)
    assert np.all(np.diff(tr.pibar) >= -1e-15)
    assert np.all(tr.pibar >= tr.pi - 1e-15)


def test_sex_specific_weights(tmp_path):
    f = simulate(write(tmp_path, sex="F", wF=1.0, wM=2.0))
    m = simulate(write(tmp_path, sex="M", wF=1.0, wM=2.0))
    assert m.T == pytest.approx(f.T / 2, abs=1e-6)


def test_genotype_has_no_phenotype_without_opportunity(tmp_path):
    # Eq. (2): dx/dz = 0 when opportunity is absent (env factor = 0)
    a = simulate(write(tmp_path, z=0, lam=3.0, envb=0.0, envc=1.0, env=0.0))
    b = simulate(write(tmp_path, z=1, lam=3.0, envb=0.0, envc=1.0, env=0.0))
    assert np.allclose(a.S, b.S)
    # ...but it does when opportunity is present
    c = simulate(write(tmp_path, z=0, lam=3.0, envb=0.0, envc=1.0, env=1.0))
    d = simulate(write(tmp_path, z=1, lam=3.0, envb=0.0, envc=1.0, env=1.0))
    assert d.S[-1] > c.S[-1]


# --------------------------------------------------------------------------- #
# Window of opportunity
# --------------------------------------------------------------------------- #

def test_intervention_after_transition_has_no_effect(tmp_path):
    p = write(tmp_path)
    base = simulate(p)
    late = simulate(p, [Intervention("d1", base.T + 5, 0.0)])
    assert late.T == pytest.approx(base.T)


def test_window_example_is_bounded():
    p = load_parameters(EXAMPLE)
    res = window_of_opportunity(p)
    assert res, "example should define windows"
    base = simulate(p)
    for r in res:
        late_starts = [pt for pt in r.points if pt.start_age >= base.T]
        assert all(not pt.in_window for pt in late_starts)
        assert any(pt.in_window for pt in r.points)


# --------------------------------------------------------------------------- #
# Example file, risk table, CLI, validation
# --------------------------------------------------------------------------- #

def test_example_runs_and_risk_increases():
    p = load_parameters(EXAMPLE)
    tr = simulate(p)
    rows = risk_by_age(tr, p.report_ages, p.risk_horizon)
    pib = [r.pibar for r in rows]
    assert all(b >= a for a, b in zip(pib, pib[1:]))
    assert 0.0 <= pib[0] <= pib[-1] <= 1.0


def test_cli_writes_outputs(tmp_path):
    rc = main([str(EXAMPLE), "--out", str(tmp_path), "--quiet"])
    assert rc == 0
    for name in ("trajectory.csv", "risk_by_age.csv", "window.csv", "summary.json"):
        assert (tmp_path / name).is_file()


def test_bad_sex_rejected(tmp_path):
    with pytest.raises(ParameterError):
        write(tmp_path, sex="X")


def test_unknown_locus_rejected(tmp_path):
    f = tmp_path / "p.param"
    f.write_text(MINIMAL.format(sex="F", env=0, z=0, beta=0, theta0=50, wF=1, wM=1, tau="inf",
                                lam=0, envb=1, envc=0, onset=0).replace("G1:0", "NOPE:1"))
    with pytest.raises(ParameterError):
        load_parameters(f)


def test_constitutive_share_narrows_window():
    # criterion (iv): W_i(k) narrows as rho_i rises
    p = load_parameters(EXAMPLE)
    q = load_parameters(EXAMPLE.parent / "parameters_psen1_carrier.param")
    tp, tq = simulate(p), simulate(q)
    assert tq.rho > tp.rho and tq.T < tp.T

    def width(params):
        res = {r.target: r for r in window_of_opportunity(params)}["metabolic"]
        return sum(pt.in_window for pt in res.points)

    assert width(q) < width(p)
