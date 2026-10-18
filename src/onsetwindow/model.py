"""Numerical implementation of the accumulating-burden model.

Equation numbers refer to the Perspective "Risk with a clock: reconsidering
Alzheimer's disease as an accumulating, whole-body process".

    (1) S_i(t) = sum_k w_k(s_i, z_i) int_0^t phi_k(t-u) x_ik(u) du + t sum_g beta_g z_ig
                 [+ kappa int_0^t m_i(u) du]
    (2) x_ik(u) = [lambda_0k + lambda_k(z_i)] e_ik(u),  e_ik(u) = e~_ik(u) 1{u >= A_ik}
    (3) x_ik(u) = (2) + nu_k m_i(u)
    (4) pi_i(t) = sigmoid(a [S_i(t) - theta_i]),  theta_i = theta_0 + gamma'r_i + eta'z_i^(theta)
        T_i = inf{t : S_i(t) >= theta_i},  pibar_i(t) = max_{u<=t} pi_i(u)
    (5) rho_i = T_i sum_g beta_g z_ig / S_i(T_i)
    (6) W_i(k) = {u : Delta T_i(k, u) > delta}

The memory kernel is exponential, phi_k(tau) = exp(-tau / tau_k); tau_k = inf
gives a permanent contribution. The integral is updated exactly for an
exposure that is constant within each time step.

The residual group-level term omega_k q_i discussed in the paper is deliberately
NOT implemented: the paper treats a non-zero omega_k as a diagnostic of
unmeasured environment, not as a biological input to individual risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .params import Domain, Parameters, Profile

try:  # scipy makes the kernel integral a single vectorised call; optional
    from scipy.signal import lfilter as _lfilter
except Exception:  # pragma: no cover - exercised only without scipy
    _lfilter = None


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def age_grid(p: Parameters) -> np.ndarray:
    n = int(round((p.age_max - p.age_start) / p.dt)) + 1
    return p.age_start + p.dt * np.arange(n)


def evaluate_profile(profile: Profile, ages: np.ndarray) -> np.ndarray:
    xp = np.asarray(profile.ages, dtype=float)
    fp = np.asarray(profile.values, dtype=float)
    if profile.interp == "linear":
        return np.interp(ages, xp, fp)
    # step: hold each value until the next breakpoint
    idx = np.searchsorted(xp, ages, side="right") - 1
    idx = np.clip(idx, 0, len(fp) - 1)
    return fp[idx]


def _dot(coefs: Dict[str, float], values: Dict[str, float]) -> float:
    return float(sum(c * values.get(k, 0.0) for k, c in coefs.items()))


def domain_weight(d: Domain, p: Parameters) -> float:
    """w_k(s_i, z_i) = w_k(s_i) * (1 + sum_g c_kg z_ig)."""
    return d.weight[p.sex] * (1.0 + _dot(d.weight_genotype, p.genotype))


def domain_onset(d: Domain, p: Parameters) -> float:
    """A_ik = A_k(z_i, E_i) = onset_age + sum_g d_kg z_ig + onset_env * E_i."""
    return d.onset_age + _dot(d.onset_genotype, p.genotype) + d.onset_env * p.environment


def domain_propensity(d: Domain, p: Parameters) -> float:
    """lambda_0k + lambda_k(z_i), with lambda_k(z_i) >= 0."""
    return d.lambda0 + max(0.0, _dot(d.lambda_genotype, p.genotype))


def threshold(p: Parameters) -> float:
    """theta_i = theta_0 + gamma' r_i + eta' z_i^(theta)."""
    return p.theta0 + _dot(p.gamma, p.reserve) + _dot(p.eta, p.reserve_genotype)


def genetic_rate(p: Parameters) -> float:
    """sum_g beta_g z_ig: constitutive burden added per year of life."""
    return _dot(p.beta, p.genotype)


def kernel_integral(x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """I(t_n) = int_0^{t_n} exp(-(t_n-u)/tau) x(u) du on a uniform grid.

    Exact when x is constant on each [t_j, t_{j+1}):
        I_{n} = d I_{n-1} + b x_{n-1},  d = exp(-dt/tau), b = tau (1 - d)
    """
    if math.isinf(tau):
        d, b = 1.0, dt
    else:
        d = math.exp(-dt / tau)
        b = tau * (1.0 - d)
    out = np.zeros_like(x, dtype=float)
    if len(x) < 2:
        return out
    if _lfilter is not None:
        out[1:] = _lfilter([b], [1.0, -d], x[:-1])
    else:  # pragma: no cover
        acc = 0.0
        for n in range(1, len(x)):
            acc = d * acc + b * x[n - 1]
            out[n] = acc
    return out


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

@dataclass
class Intervention:
    """Scale the exposure of one target from ``start_age`` onwards.

    ``target`` is a domain name or ``"latent"`` (which scales m_i(u) and so
    acts on every domain that loads on it, and on the kappa term).
    ``scale`` < 1 reduces the exposure; for a protective domain (w_k < 0)
    use ``scale`` > 1 to strengthen it.
    """

    target: str
    start_age: float
    scale: float


@dataclass
class Trajectory:
    ages: np.ndarray
    S: np.ndarray
    theta: float
    pi: np.ndarray
    pibar: np.ndarray
    T: Optional[float]                    # transition age; None if no crossing by age_max
    rho: Optional[float]
    contributions: Dict[str, np.ndarray]  # per-domain w_k * integral, plus genetic, latent
    exposures: Dict[str, np.ndarray]      # x_ik(u) per domain
    weights: Dict[str, float]
    onsets: Dict[str, float]
    latent: np.ndarray
    interventions: List[Intervention] = field(default_factory=list)

    def at(self, series: np.ndarray, age: float) -> float:
        return float(np.interp(age, self.ages, series))


def _first_crossing(ages: np.ndarray, S: np.ndarray, theta: float) -> Optional[float]:
    hit = np.nonzero(S >= theta)[0]
    if hit.size == 0:
        return None
    n = int(hit[0])
    if n == 0:
        return float(ages[0])
    s0, s1 = S[n - 1], S[n]
    frac = (theta - s0) / (s1 - s0) if s1 != s0 else 1.0
    return float(ages[n - 1] + frac * (ages[n] - ages[n - 1]))


def simulate(p: Parameters, interventions: Optional[List[Intervention]] = None) -> Trajectory:
    """Run the model for one individual and return the full trajectory."""
    interventions = list(interventions or [])
    ages = age_grid(p)
    dt = p.dt

    # latent deficit m_i(u), optionally modified by an intervention on 'latent'
    m = evaluate_profile(p.latent, ages)
    for iv in interventions:
        if iv.target == "latent":
            m = np.where(ages >= iv.start_age, m * iv.scale, m)

    S = np.zeros_like(ages)
    contributions: Dict[str, np.ndarray] = {}
    exposures: Dict[str, np.ndarray] = {}
    weights: Dict[str, float] = {}
    onsets: Dict[str, float] = {}

    env = p.environment
    for d in p.domains:
        A = domain_onset(d, p)
        opportunity = evaluate_profile(d.opportunity, ages) * max(0.0, d.env_baseline + d.env_coef * env)
        e = opportunity * (ages >= A)                                   # Eq. (2)
        x = domain_propensity(d, p) * e + d.nu * m                       # Eq. (3)
        for iv in interventions:
            if iv.target == d.name:
                x = np.where(ages >= iv.start_age, x * iv.scale, x)
        w = domain_weight(d, p)
        contrib = w * kernel_integral(x, d.tau, dt)                     # Eq. (1), domain term
        S = S + contrib
        contributions[d.name] = contrib
        exposures[d.name] = x
        weights[d.name] = w
        onsets[d.name] = A

    G = genetic_rate(p)
    genetic = ages * G                                                    # t * sum beta z
    S = S + genetic
    contributions["genetic"] = genetic

    if p.kappa > 0:
        latent_direct = p.kappa * kernel_integral(m, math.inf, dt)       # kappa int m du
        S = S + latent_direct
        contributions["latent_direct"] = latent_direct

    theta = threshold(p)
    pi = 1.0 / (1.0 + np.exp(-p.a * (S - theta)))                        # Eq. (4)
    pibar = np.maximum.accumulate(pi)                                    # absorbing transition
    T = _first_crossing(ages, S, theta)
    rho = None
    if T is not None and theta != 0:
        rho = T * G / theta                                              # Eq. (5), S(T) = theta

    return Trajectory(
        ages=ages, S=S, theta=theta, pi=pi, pibar=pibar, T=T, rho=rho,
        contributions=contributions, exposures=exposures, weights=weights,
        onsets=onsets, latent=m, interventions=interventions,
    )


# --------------------------------------------------------------------------- #
# Risk at given ages
# --------------------------------------------------------------------------- #

@dataclass
class RiskRow:
    age: float
    burden: float
    pi: float
    pibar: float
    horizon_risk: Optional[float]   # P(manifest by age+h | not manifest by age)


def risk_by_age(traj: Trajectory, ages: List[float], horizon: float) -> List[RiskRow]:
    rows = []
    a_max = float(traj.ages[-1])
    for age in ages:
        if age < traj.ages[0] or age > a_max:
            raise ValueError(f"report age {age} is outside the simulated range")
        pb = traj.at(traj.pibar, age)
        hr = None
        if horizon > 0 and age + horizon <= a_max + 1e-9 and pb < 1.0:
            hr = (traj.at(traj.pibar, age + horizon) - pb) / (1.0 - pb)
        rows.append(RiskRow(age=age, burden=traj.at(traj.S, age), pi=traj.at(traj.pi, age),
                            pibar=pb, horizon_risk=hr))
    return rows


# --------------------------------------------------------------------------- #
# Window of opportunity, Eq. (6)
# --------------------------------------------------------------------------- #

@dataclass
class WindowPoint:
    start_age: float
    T_new: Optional[float]
    delta_T: Optional[float]
    censored: bool                  # True: no crossing by age_max, delta_T is a lower bound
    in_window: bool


@dataclass
class WindowResult:
    target: str
    scale: float
    delta: float
    points: List[WindowPoint]
    intervals: List[Tuple[float, float]]


def _intervals(starts: List[float], flags: List[bool], step: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    run_start = None
    prev = None
    for s, f in zip(starts, flags):
        if f and run_start is None:
            run_start = s
        if not f and run_start is not None:
            out.append((run_start, prev))
            run_start = None
        prev = s
    if run_start is not None:
        out.append((run_start, prev))
    return out


def window_of_opportunity(p: Parameters, base: Optional[Trajectory] = None) -> List[WindowResult]:
    """Compute W_i(k) for every target in the [window] section.

    Returns an empty list if there is no [window] section or the individual
    does not reach the threshold by age_max (T_i undefined).
    """
    if p.window is None:
        return []
    base = base or simulate(p)
    if base.T is None:
        return []
    ws = p.window
    a_max = float(base.ages[-1])
    step = ws.start_ages[1] - ws.start_ages[0] if len(ws.start_ages) > 1 else 1.0
    results: List[WindowResult] = []
    for target in ws.targets:
        scale = ws.scales[target]
        points: List[WindowPoint] = []
        for u in ws.start_ages:
            traj = simulate(p, [Intervention(target, u, scale)])
            if traj.T is None:
                dT, censored = a_max - base.T, True
            else:
                dT, censored = traj.T - base.T, False
            points.append(WindowPoint(start_age=u, T_new=traj.T, delta_T=dT,
                                      censored=censored, in_window=dT > ws.delta))
        intervals = _intervals([pt.start_age for pt in points], [pt.in_window for pt in points], step)
        results.append(WindowResult(target=target, scale=scale, delta=ws.delta,
                                    points=points, intervals=intervals))
    return results
