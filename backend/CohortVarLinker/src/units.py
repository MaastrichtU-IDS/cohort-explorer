"""units.py — Deterministic unit comparison for continuous variables.

Decides how two study units relate, using pint for the arithmetic so no
conversion factor is hand-maintained:

  equivalent            same quantity, factor 1        (ng/mL vs ug/L, 10*3/uL vs /nL)
  scalable              same dimension, fixed factor   (g/L vs g/dL, U/L vs ukat/L)
  molar_mass_dependent  amount vs mass of substance    (mmol/L vs mg/dL) — needs the
                        analyte's molar mass, which the units alone cannot supply
  incommensurable       different dimensions           (% target dose vs mg, mL/m2 vs mL)
  unparseable           not sure: not a unit, not understood, or not safe to read
                        (DD-MM-YYYY, {score}, degC, MG — megagauss or milligram?)

The rule throughout: give a factor only when the reading is unambiguous, and
otherwise say "not sure" so the pair is left to the semantic check. A
confident wrong factor is worse than none.

Study dictionaries write UCUM loosely (``ucum:`` prefixes, lower-cased codes,
``m2`` for m^2, ``{annotations}``, ``1.73m2``), so each string is normalised to
pint syntax first. Unit strings come from data dictionaries, i.e. untrusted
input: they are length-limited and character-whitelisted before the parser.

``check_ranges`` tests a factor against the value ranges each dictionary
records, which catches unit labels that do not match the data (a hemoglobin
labelled g/dL whose values run 75–178).
"""
from __future__ import annotations

import math
import re
import threading
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Tuple

# pint is pinned with hashes in poetry.lock. Fail loudly rather than degrade: a
# silent fallback would change harmonization verdicts without anyone noticing.
try:
    import pint
except ImportError as exc:  # pragma: no cover - exercised only on broken installs
    raise ImportError(
        "src.units requires 'pint' (locked at 0.24.4 in poetry.lock). "
        "Install the locked environment with `poetry install`."
    ) from exc

_MIN_PINT = (0, 24)
_pint_version = tuple(int(p) for p in re.findall(r"\d+", pint.__version__)[:2])
if _pint_version < _MIN_PINT:  # pragma: no cover
    raise ImportError(
        f"src.units requires pint>={'.'.join(map(str, _MIN_PINT))}, found {pint.__version__}."
    )


EQUIVALENT = "equivalent"
SCALABLE = "scalable"
MOLAR_MASS_DEPENDENT = "molar_mass_dependent"
INCOMMENSURABLE = "incommensurable"
UNPARSEABLE = "unparseable"

# Outcomes of check_ranges.
RANGE_CONFIRMED = "confirmed"
RANGE_CONTRADICTED = "contradicted"
RANGE_UNVERIFIED = "unverified"

_FACTOR_REL_TOL = 1e-9
_MAX_UNIT_LEN = 64
# Anything outside this set never reaches pint.
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9/*.^()\[\]{}_%+\-]*$")
# Date/time formats written in the units column (DD-MM-YYYY, hh/mm). Whole
# tokens only, so a real unit such as mm/d (millimetres per day) is not caught.
_DATE_FORMAT = re.compile(r"^(dd|mm|yy|yyyy|hh|ss)([-/.:](dd|mm|yy|yyyy|hh|ss))+$", re.IGNORECASE)

# Bracketed UCUM codes pint reads differently.
_BRACKET_ALIASES = {
    "[u]": "U",              # enzyme unit, umol/min
    "[iu]": "intl_unit",     # international unit: analyte-specific, only comparable to itself
}
_SYMBOL_ALIASES = {
    "u": "U",                # pint's "u" is the dalton; in a clinical unit it is the enzyme unit
    "mo": "month",           # UCUM mean Julian month; pint has no 'mo'
    "wk": "week",            # UCUM week; pint has no 'wk'
    "deg": "angle_degree",   # pint angles are dimensionless and would scale to %
    "degree": "angle_degree",
    "degrees": "angle_degree",
    "beats": "1",
    "beat": "1",
    "bpm": "(1/min)",
    "IU": "intl_unit",
    "iu": "intl_unit",
}
# Words pint resolves to units that do not occur as clinical measurement
# units: reading them would produce a confident wrong comparison.
#   t -> tonne (a lower-cased T/L, i.e. 10*12/L), unit -> micro-nit (luminance),
#   point/pt -> typographic point / pint, grade -> angle.
_NOT_UNITS = frozenset({"t", "unit", "units", "point", "points", "pt", "grade", "grades"})
# pint joins any prefix to any unit, so non-units read as real ones ("hh",
# an hours placeholder, is a hecto-hour). Clinical units use neither these
# prefixes nor prefixed calendar units, so such a reading is rejected.
_UNUSED_PREFIXES = frozenset({"hecto", "deca", "deka", "exa", "peta", "zetta", "yotta",
                              "atto", "zepto", "yocto", "quecto", "ronto", "ronna", "quetta"})
_UNPREFIXABLE = frozenset({"minute", "hour", "day", "week", "month", "year"})


@dataclass(frozen=True)
class UnitComparison:
    """How a value in ``source_unit`` relates to one in ``target_unit``.

    ``factor`` multiplies a source value to express it in the target unit.
    For ``molar_mass_dependent`` it is the factor per kg/mol of molar mass;
    use :meth:`factor_with_molar_mass`.
    """
    relation: str
    factor: Optional[float] = None
    reason: str = ""
    source_is_substance: Optional[bool] = None

    @property
    def is_deterministic(self) -> bool:
        return self.relation in (EQUIVALENT, SCALABLE)

    def factor_with_molar_mass(self, molar_mass_g_per_mol: float) -> Optional[float]:
        if self.relation != MOLAR_MASS_DEPENDENT or self.factor is None:
            return None
        if not molar_mass_g_per_mol or molar_mass_g_per_mol <= 0:
            return None
        kg_per_mol = molar_mass_g_per_mol / 1000.0
        return self.factor * kg_per_mol if self.source_is_substance else self.factor / kg_per_mol


# ── Registry ────────────────────────────────────────────────────────────────
# One private registry (never pint's shared application registry), no disk
# cache, no definition files. Phase A runs worker threads, so creation and
# every registry call are serialised.

_REGISTRY: Optional["pint.UnitRegistry"] = None
_REGISTRY_LOCK = threading.Lock()


def _registry() -> "pint.UnitRegistry":
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                ureg = pint.UnitRegistry(cache_folder=None)
                ureg.define("intl_unit = [international_unit]")
                ureg.define("angle_degree = [plane_angle]")
                _REGISTRY = ureg
    return _REGISTRY


# ── Normalisation: loose UCUM -> pint expression ────────────────────────────

def _map_symbol(symbol: str, ureg: "pint.UnitRegistry") -> Optional[str]:
    """pint name for one unit symbol, or None when it must not be read."""
    if symbol.startswith("["):
        return _BRACKET_ALIASES.get(symbol.lower(), symbol)
    if symbol in _SYMBOL_ALIASES:
        return _SYMBOL_ALIASES[symbol]
    if symbol in _NOT_UNITS:
        return None
    # A capitalised word (Kilogram, Years) is a unit name, not a symbol whose
    # case matters; symbols this short (Mg, Pa) keep their case.
    if len(symbol) >= 4 and symbol.istitle():
        symbol = symbol.lower()
    with _REGISTRY_LOCK:
        if symbol in ureg:
            return symbol
        # Dictionaries lower-case UCUM codes (mV -> mv, mm[Hg] -> mmhg). Only a
        # lower-case symbol is looked up without case, and only a single
        # candidate is accepted: upper-case input is where case carries meaning.
        # Single letters are excluded: w (watt? week?), s, g, t change meaning
        # with case, and a lone letter gives nothing else to go on.
        if symbol.islower() and len(symbol) > 1:
            candidates = ureg.parse_unit_name(symbol, case_sensitive=False)
            if len(candidates) == 1:
                prefix, name, suffix = candidates[0]
                return f"{prefix}{name}{suffix}"
    return symbol   # unknown: pint rejects it and the unit is unparseable


def _coefficient_term(m: "re.Match") -> str:
    """'1.73m2' -> '(173/100*m2)': a number written against a unit is one term."""
    whole, frac, sym, exp = m.group(1), m.group(2) or "", m.group(3), m.group(4) or ""
    return f"({whole}{frac}/{10 ** len(frac)}*{sym}{exp})"


def to_pint_expression(raw: Optional[str]) -> Optional[str]:
    """Normalise a dictionary unit string to a pint expression, or None."""
    if raw is None:
        return None
    s = str(raw).replace("µ", "u").replace("μ", "u")   # micro sign, Greek mu
    s = re.sub(r"^\s*per\s+", "1/", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+per\s+", "/", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^ucum:", "", s, flags=re.IGNORECASE)
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    if len(s) > _MAX_UNIT_LEN or not _ALLOWED_CHARS.match(s):
        return None
    if _DATE_FORMAT.match(s):
        return None
    # A bare count as numerator is how a lost superscript looks: 109/L is
    # 10^9/L, 1012/L is 10^12/L. Only "1/..." is read as written.
    lead = re.match(r"^\(?(\d+)\)?/", s)
    if lead and lead.group(1) != "1":
        return None
    # An annotation alone ({score}, {ratio}) labels a quantity without saying
    # what scale it is on, so there is nothing to compare.
    if "{" in s and re.sub(r"\{[^}]*\}", "", s) in ("", "1"):
        return None

    # Body-surface normalisation, written two ways. Kept as a real divisor so
    # mL/min/1.73m2 never equals plain mL/min.
    s = s.replace("{1.73_m2}", "1.73m2")
    s = re.sub(r"\{[^}]*\}", "1", s)                          # other annotations: {counts}, {cell}
    s = re.sub(r"mm\[hg\]", "mmHg", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<!\d)10[*^](-?\d+)", r"(10**\1)", s)        # UCUM powers: 10*3 -> (10**3)
    s = re.sub(r"(?<![A-Za-z0-9_)\]])(\d+)(?:\.(\d+))?([A-Za-z]+)(-?\d+)?", _coefficient_term, s)
    s = s.replace(".", "*")                                    # UCUM multiplication
    if s.startswith("/"):
        s = "1" + s

    ureg = _registry()
    rejected = []

    def _symbol(m: "re.Match") -> str:
        sym, exp = m.group(1), m.group(2)
        mapped = _map_symbol(sym, ureg)
        if mapped is None:
            rejected.append(sym)
            return sym
        return f"{mapped}**{exp}" if exp else mapped

    # A unit symbol, optionally followed by a UCUM exponent (m2, cm-1).
    s = re.sub(r"(?<![\d.])(\[[^\]]*\]|[A-Za-z%]+)(-?\d+)?", _symbol, s)
    return None if rejected else s


@lru_cache(maxsize=2048)
def _parse(expr: str) -> Optional[Tuple[float, FrozenSet[Tuple[str, float]]]]:
    """(magnitude in base units, dimensionality) for a pint expression."""
    ureg = _registry()
    try:
        with _REGISTRY_LOCK:
            q = ureg.parse_expression(expr)
            if not isinstance(q, ureg.Quantity):
                q = ureg.Quantity(q)
            # Offset scales (degC, degF) are not related by a factor.
            # (pint 0.24 API; tests.test_units pins the behaviour.)
            if not getattr(q, "_is_multiplicative", True):
                return None
            for name in q.units._units:
                candidates = ureg.parse_unit_name(name)
                if not candidates:
                    continue
                prefix, unit_name, _ = candidates[0]
                if prefix in _UNUSED_PREFIXES or (prefix and unit_name in _UNPREFIXABLE):
                    return None
            base = q.to_base_units()
            magnitude = float(base.magnitude)
            dims = frozenset((str(k), float(v)) for k, v in base.dimensionality.items())
    # pint runs the string through Python's tokenizer (never eval), so
    # malformed input surfaces as TokenError as well as pint's own errors.
    except (pint.PintError, tokenize.TokenError, AssertionError, AttributeError,
            IndexError, KeyError, TypeError, ValueError, ZeroDivisionError,
            SyntaxError, OverflowError):
        return None
    if not math.isfinite(magnitude) or magnitude <= 0:
        return None
    return magnitude, dims


def _same(a, b) -> bool:
    return a[1] == b[1] and math.isclose(a[0], b[0], rel_tol=_FACTOR_REL_TOL)


def parse_unit(raw: Optional[str]) -> Optional[Tuple[float, FrozenSet[Tuple[str, float]]]]:
    """Parse a dictionary unit string, or None when not sure what it means.

    A unit written with capitals must mean the same in lower case: mg/dL and
    mg/dl agree, but MG (megagauss) and mg do not, nor ML (megalitre) and mL,
    nor G/L (gauss per litre) and g/L. Where case changes the meaning the
    string is not read at all.
    """
    expr = to_pint_expression(raw)
    parsed = _parse(expr) if expr else None
    if parsed is None:
        return None
    text = str(raw)
    if text != text.lower():
        lower_expr = to_pint_expression(text.lower())
        lower = _parse(lower_expr) if lower_expr else None
        if lower is None or not _same(parsed, lower):
            return None
    return parsed


def _dim_ratio(a: FrozenSet[Tuple[str, float]], b: FrozenSet[Tuple[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = dict(a)
    for k, v in b:
        out[k] = out.get(k, 0.0) - v
    return {k: v for k, v in out.items() if v != 0}


def compare_units(source_unit: Optional[str], target_unit: Optional[str]) -> UnitComparison:
    """Classify the relation between two dictionary unit strings."""
    src, tgt = parse_unit(source_unit), parse_unit(target_unit)
    if src is None or tgt is None:
        bad = source_unit if src is None else target_unit
        return UnitComparison(UNPARSEABLE, reason=f"unit not understood: {bad!r}")

    (src_mag, src_dims), (tgt_mag, tgt_dims) = src, tgt
    factor = src_mag / tgt_mag

    if src_dims == tgt_dims:
        if math.isclose(factor, 1.0, rel_tol=_FACTOR_REL_TOL):
            return UnitComparison(EQUIVALENT, 1.0, f"{source_unit} equals {target_unit}")
        return UnitComparison(SCALABLE, factor, f"{source_unit} x {factor:g} = {target_unit}")

    ratio = _dim_ratio(src_dims, tgt_dims)
    if ratio == {"[substance]": 1.0, "[mass]": -1.0}:
        return UnitComparison(MOLAR_MASS_DEPENDENT, factor,
                              "amount-of-substance vs mass: needs the analyte's molar mass",
                              source_is_substance=True)
    if ratio == {"[mass]": 1.0, "[substance]": -1.0}:
        return UnitComparison(MOLAR_MASS_DEPENDENT, factor,
                              "mass vs amount-of-substance: needs the analyte's molar mass",
                              source_is_substance=False)

    return UnitComparison(INCOMMENSURABLE,
                          reason=f"different dimensions: {source_unit} vs {target_unit}")


# ── Value-range check ───────────────────────────────────────────────────────

def _valid_range(lo, hi) -> bool:
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lo) and math.isfinite(hi) and lo <= hi


def check_ranges(factor: Optional[float], src_min, src_max, tgt_min, tgt_max) -> str:
    """Do the recorded value ranges agree with converting source by ``factor``?

    The source range, converted, must overlap the target range. Min/max hold
    outliers, so this only catches gross disagreements (x10, x100, x1000: a
    fraction labelled %, g/L labelled g/dL); it cannot certify a factor. Returns
    "unverified" when a range is missing or unusable.
    """
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return RANGE_UNVERIFIED
    if not math.isfinite(factor) or factor <= 0:
        return RANGE_UNVERIFIED
    if not (_valid_range(src_min, src_max) and _valid_range(tgt_min, tgt_max)):
        return RANGE_UNVERIFIED
    lo, hi = float(src_min) * factor, float(src_max) * factor
    t_lo, t_hi = float(tgt_min), float(tgt_max)
    slack = _FACTOR_REL_TOL * max(abs(hi), abs(t_hi), 1.0)
    return RANGE_CONFIRMED if lo <= t_hi + slack and t_lo <= hi + slack else RANGE_CONTRADICTED
