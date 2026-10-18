"""Read and validate a ``.param`` file.

The file is INI-style (sections in square brackets, ``key = value`` lines,
``#`` or ``;`` for comments). See ``examples/parameters.param`` for a fully
commented example and ``README.md`` for the meaning of every key.
"""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


class ParameterError(ValueError):
    """Raised when the parameter file is missing a key or holds an invalid value."""


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #

def _float(value: str, where: str) -> float:
    v = value.strip().lower()
    if v in ("inf", "infinity", "+inf"):
        return math.inf
    try:
        return float(v)
    except ValueError as exc:
        raise ParameterError(f"{where}: expected a number, got '{value}'") from exc


def _float_list(value: str, where: str) -> List[float]:
    items = [s for s in (p.strip() for p in value.split(",")) if s]
    if not items:
        raise ParameterError(f"{where}: expected a comma-separated list of numbers")
    return [_float(s, where) for s in items]


def _str_list(value: str) -> List[str]:
    return [s for s in (p.strip() for p in value.split(",")) if s]


def _mapping(value: str, where: str) -> Dict[str, float]:
    """Parse ``LOCUS:coef, LOCUS:coef`` into a dict. Empty string gives {}."""
    out: Dict[str, float] = {}
    for item in _str_list(value):
        if ":" not in item:
            raise ParameterError(f"{where}: expected 'name:value' pairs, got '{item}'")
        name, num = item.split(":", 1)
        out[name.strip()] = _float(num, where)
    return out


def _section_numbers(cp: configparser.ConfigParser, name: str) -> Dict[str, float]:
    if not cp.has_section(name):
        return {}
    return {k: _float(v, f"[{name}] {k}") for k, v in cp.items(name)}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class Profile:
    """A piecewise function of age, given by breakpoints.

    Outside the breakpoints the first/last value is held constant.
    ``interp`` is ``linear`` (straight lines between breakpoints) or ``step``
    (value holds until the next breakpoint, so changes are abrupt).
    """

    ages: List[float]
    values: List[float]
    interp: str = "linear"

    def validate(self, where: str) -> None:
        if len(self.ages) != len(self.values):
            raise ParameterError(f"{where}: ages and values must have the same length")
        if any(b <= a for a, b in zip(self.ages, self.ages[1:])):
            raise ParameterError(f"{where}: ages must be strictly increasing")
        if self.interp not in ("linear", "step"):
            raise ParameterError(f"{where}: interpolation must be 'linear' or 'step'")


@dataclass
class Domain:
    """One risk domain k (metabolic, vascular, immune, environmental, protective ...)."""

    name: str
    weight: Dict[str, float]               # sex -> w_k(s); sign allowed (w<0 = protective)
    weight_genotype: Dict[str, float]      # multiplicative modifier: w = w_s * (1 + sum c_g z_g)
    tau: float                             # kernel persistence (years); inf = permanent
    lambda0: float                         # baseline propensity lambda_0k
    lambda_genotype: Dict[str, float]      # lambda_k(z) = max(0, sum l_g z_g)
    opportunity: Profile                   # base profile of e~_ik(u)
    env_baseline: float                    # e~ = profile * (env_baseline + env_coef * E)
    env_coef: float
    onset_age: float                       # A_k = onset_age + sum d_g z_g + onset_env * E
    onset_genotype: Dict[str, float]
    onset_env: float
    nu: float                              # loading on latent deficit m_i(u), nu_k >= 0


@dataclass
class WindowSpec:
    targets: List[str]
    scales: Dict[str, float]
    delta: float
    start_ages: List[float]


@dataclass
class Parameters:
    # simulation
    age_start: float
    age_max: float
    dt: float
    report_ages: List[float]
    risk_horizon: float
    # individual
    individual_id: str
    sex: str
    environment: float
    genotype: Dict[str, float]
    reserve: Dict[str, float]
    reserve_genotype: Dict[str, float]
    # population-level coefficients
    beta: Dict[str, float]
    gamma: Dict[str, float]
    eta: Dict[str, float]
    theta0: float
    a: float
    latent: Profile
    kappa: float
    domains: List[Domain]
    window: Optional[WindowSpec] = None
    source: Optional[str] = field(default=None, repr=False)

    def domain(self, name: str) -> Domain:
        for d in self.domains:
            if d.name == name:
                return d
        raise KeyError(name)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def _require(cp: configparser.ConfigParser, section: str, key: str) -> str:
    if not cp.has_section(section):
        raise ParameterError(f"missing section [{section}]")
    if not cp.has_option(section, key):
        raise ParameterError(f"missing key '{key}' in [{section}]")
    return cp.get(section, key)


def _get(cp, section, key, default):
    if cp.has_section(section) and cp.has_option(section, key):
        return cp.get(section, key)
    return default


def load_parameters(path: str | Path) -> Parameters:
    """Read a ``.param`` file and return a validated :class:`Parameters`."""
    path = Path(path)
    if not path.is_file():
        raise ParameterError(f"parameter file not found: {path}")

    cp = configparser.ConfigParser(
        inline_comment_prefixes=("#", ";"), comment_prefixes=("#", ";"),
        interpolation=None,
    )
    cp.optionxform = str  # keep the case of locus and domain names
    cp.read(path, encoding="utf-8")

    # ---- simulation ------------------------------------------------------- #
    age_start = _float(_get(cp, "simulation", "age_start", "0"), "[simulation] age_start")
    age_max = _float(_require(cp, "simulation", "age_max"), "[simulation] age_max")
    dt = _float(_get(cp, "simulation", "dt", "0.1"), "[simulation] dt")
    report_ages = _float_list(_require(cp, "simulation", "report_ages"), "[simulation] report_ages")
    risk_horizon = _float(_get(cp, "simulation", "risk_horizon", "5"), "[simulation] risk_horizon")

    # ---- individual ------------------------------------------------------- #
    individual_id = _get(cp, "individual", "id", "individual").strip()
    sex = _require(cp, "individual", "sex").strip().upper()
    environment = _float(_get(cp, "individual", "environment", "0"), "[individual] environment")

    genotype = _section_numbers(cp, "genotype")
    reserve = _section_numbers(cp, "reserve")
    reserve_genotype = _section_numbers(cp, "reserve_genotype")

    # ---- coefficients ----------------------------------------------------- #
    beta = _section_numbers(cp, "beta")
    gamma = _section_numbers(cp, "gamma")
    eta = _section_numbers(cp, "eta")
    theta0 = _float(_require(cp, "threshold", "theta0"), "[threshold] theta0")
    a = _float(_require(cp, "threshold", "a"), "[threshold] a")

    latent = Profile(
        ages=_float_list(_get(cp, "latent", "ages", "0"), "[latent] ages"),
        values=_float_list(_get(cp, "latent", "values", "0"), "[latent] values"),
        interp=_get(cp, "latent", "interpolation", "linear").strip().lower(),
    )
    kappa = _float(_get(cp, "latent", "kappa", "0"), "[latent] kappa")

    # ---- domains ---------------------------------------------------------- #
    domains: List[Domain] = []
    for sec in cp.sections():
        if not sec.lower().startswith("domain:"):
            continue
        name = sec.split(":", 1)[1].strip()
        w = f"[{sec}]"
        weight = {
            "F": _float(_require(cp, sec, "weight_F"), f"{w} weight_F"),
            "M": _float(_require(cp, sec, "weight_M"), f"{w} weight_M"),
        }
        domains.append(Domain(
            name=name,
            weight=weight,
            weight_genotype=_mapping(_get(cp, sec, "weight_genotype", ""), f"{w} weight_genotype"),
            tau=_float(_require(cp, sec, "tau"), f"{w} tau"),
            lambda0=_float(_get(cp, sec, "lambda0", "1"), f"{w} lambda0"),
            lambda_genotype=_mapping(_get(cp, sec, "lambda_genotype", ""), f"{w} lambda_genotype"),
            opportunity=Profile(
                ages=_float_list(_require(cp, sec, "opportunity_ages"), f"{w} opportunity_ages"),
                values=_float_list(_require(cp, sec, "opportunity_values"), f"{w} opportunity_values"),
                interp=_get(cp, sec, "opportunity_interpolation", "linear").strip().lower(),
            ),
            env_baseline=_float(_get(cp, sec, "env_baseline", "1"), f"{w} env_baseline"),
            env_coef=_float(_get(cp, sec, "env_coef", "0"), f"{w} env_coef"),
            onset_age=_float(_get(cp, sec, "onset_age", "0"), f"{w} onset_age"),
            onset_genotype=_mapping(_get(cp, sec, "onset_genotype", ""), f"{w} onset_genotype"),
            onset_env=_float(_get(cp, sec, "onset_env", "0"), f"{w} onset_env"),
            nu=_float(_get(cp, sec, "nu", "0"), f"{w} nu"),
        ))

    # ---- window (optional) ------------------------------------------------ #
    window = None
    if cp.has_section("window"):
        targets = _str_list(_require(cp, "window", "targets"))
        scale_list = _float_list(_require(cp, "window", "scale"), "[window] scale")
        if len(scale_list) == 1:
            scale_list = scale_list * len(targets)
        if len(scale_list) != len(targets):
            raise ParameterError("[window] scale must have one value, or one per target")
        start_min = _float(_require(cp, "window", "start_min"), "[window] start_min")
        start_max = _float(_require(cp, "window", "start_max"), "[window] start_max")
        start_step = _float(_get(cp, "window", "start_step", "1"), "[window] start_step")
        if start_step <= 0 or start_max < start_min:
            raise ParameterError("[window] needs start_min <= start_max and start_step > 0")
        n = int(math.floor((start_max - start_min) / start_step + 1e-9)) + 1
        window = WindowSpec(
            targets=targets,
            scales=dict(zip(targets, scale_list)),
            delta=_float(_require(cp, "window", "delta"), "[window] delta"),
            start_ages=[start_min + i * start_step for i in range(n)],
        )

    params = Parameters(
        age_start=age_start, age_max=age_max, dt=dt, report_ages=report_ages,
        risk_horizon=risk_horizon, individual_id=individual_id, sex=sex,
        environment=environment, genotype=genotype, reserve=reserve,
        reserve_genotype=reserve_genotype, beta=beta, gamma=gamma, eta=eta,
        theta0=theta0, a=a, latent=latent, kappa=kappa, domains=domains,
        window=window, source=str(path),
    )
    validate(params)
    return params


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(p: Parameters) -> None:
    """Check internal consistency; raise :class:`ParameterError` on the first problem."""
    if p.sex not in ("F", "M"):
        raise ParameterError("[individual] sex must be F or M")
    if p.dt <= 0:
        raise ParameterError("[simulation] dt must be positive")
    if p.age_max <= p.age_start:
        raise ParameterError("[simulation] age_max must exceed age_start")
    if p.a <= 0:
        raise ParameterError("[threshold] a must be positive")
    if p.kappa < 0:
        raise ParameterError("[latent] kappa must be >= 0")
    if not p.domains:
        raise ParameterError("define at least one [domain:NAME] section")
    p.latent.validate("[latent]")
    if any(v < 0 for v in p.latent.values):
        raise ParameterError("[latent] values must be >= 0 (m_i(t) >= 0)")

    loci = set(p.genotype)
    for name in p.beta:
        if name not in loci:
            raise ParameterError(f"[beta] {name}: locus not listed in [genotype]")
    for name in p.gamma:
        if name not in p.reserve:
            raise ParameterError(f"[gamma] {name}: proxy not listed in [reserve]")
    for name in p.eta:
        if name not in p.reserve_genotype:
            raise ParameterError(f"[eta] {name}: locus not listed in [reserve_genotype]")

    names = [d.name for d in p.domains]
    if len(set(names)) != len(names):
        raise ParameterError("domain names must be unique")
    for d in p.domains:
        w = f"[domain:{d.name}]"
        if not (d.tau > 0):
            raise ParameterError(f"{w} tau must be > 0 (use 'inf' for a permanent contribution)")
        if d.nu < 0:
            raise ParameterError(f"{w} nu must be >= 0")
        if d.lambda0 < 0:
            raise ParameterError(f"{w} lambda0 must be >= 0")
        d.opportunity.validate(w)
        if any(v < 0 for v in d.opportunity.values):
            raise ParameterError(f"{w} opportunity_values must be >= 0")
        for field_name in ("weight_genotype", "lambda_genotype", "onset_genotype"):
            for locus in getattr(d, field_name):
                if locus not in loci:
                    raise ParameterError(f"{w} {field_name}: locus '{locus}' not listed in [genotype]")

    if p.window is not None:
        for t in p.window.targets:
            if t != "latent" and t not in names:
                raise ParameterError(f"[window] target '{t}' is neither a domain nor 'latent'")
        if p.window.delta < 0:
            raise ParameterError("[window] delta must be >= 0")
