"""Grid carbon intensity presets and their forward projection.

Separated from `carbon_model.py` because intensity is an input the model is
parameterised over, not part of the break-even arithmetic. Phase 9 sweeps it and
Phase 7 sources it; this module is where both meet the model.

**Sourced 2026-08-23 from EPA eGRID2023.** The four placeholder numbers that
came from an uncited spec draft are withdrawn. Every preset now carries a
`kg_co2e_per_kwh` read off eGRID's Table 1, `sourced=True` and a citation naming
the vintage; those three are set together, since the flag is not a thing to flip
on its own.

Three choices had to be made before a number could be looked up at all, and each
is stated here rather than left implicit:

**CO2e, not CO2.** Embodied carbon is in kg CO2e, so the operational side has to
match or the two halves of the inequality measure different things. eGRID
publishes both; the CO2e column is used. That is also why the field is
`kg_co2e_per_kwh` and not `kg_co2_per_kwh`.

**Average, not marginal.** A replacement changes load at the margin, so a
marginal rate is arguably the more correct choice. Total output rates are used
anyway, for two reasons. They are what eGRID publishes as its headline figure
and what the comparable literature uses, and they are **conservative**: eGRID's
non-baseload rate, its own rough proxy for marginal generation, is roughly twice
the total output rate in every region here, so using it would roughly halve
every payback threshold. `NONBASELOAD` carries those figures for the Phase 9
sensitivity arm rather than hiding the choice.

**Generation at the busbar, not consumption at the plug.** Output emission rates
are measured where the generator meets the grid. A GPU draws at the far end of
transmission and distribution, so the generation actually caused is higher by
the grid gross loss, which eGRID reports at 4.1 to 4.2% for these regions. That
correction is not applied. It understates avoided carbon by about 4%, which is
the same direction as the other two choices.

Declining intensity is modelled as a constant fractional decline per year, which
is the simplest form that captures the direction the grid is actually moving.
It matters because it works against replacement: every future year of savings
avoids less carbon than the year before, so the payback horizon stretches. A
model that held intensity constant would quietly favour replacing.
"""

from dataclasses import dataclass

# Rough annual decarbonisation rate for a US grid region, as a fraction per
# year. **Still unsourced**, and the only unsourced number left in this module.
# It is not used by default: GridIntensity.annual_decline defaults to 0.0, so
# the shipped presets are a flat present-day snapshot and stay fully sourced.
#
# Phase 9 owns replacing it, and the source to use is NREL Cambium, which
# publishes projected annual and hourly emission factors to 2050 under several
# scenarios, in both average and long-run marginal forms. Fitting a decline rate
# to a Cambium trajectory for the chosen region gives a cited number for this
# constant and for the marginal arm at the same time.
# https://www.nrel.gov/analysis/cambium.html
EXAMPLE_ANNUAL_DECLINE = 0.03


@dataclass(frozen=True)
class GridIntensity:
    """Carbon intensity of a grid region, optionally declining over time.

    Attributes:
      name: Region label, used in output.
      kg_co2e_per_kwh: Intensity in year zero.
      sourced: False until a citation is attached. Taints every figure computed
        from it, which is the point.
      citation: Where the number came from. Required when sourced is True.
      annual_decline: Fractional decline per year. 0.0 holds intensity constant,
        which makes the snapshot form a degenerate case of the integral rather
        than a separate code path.
    """

    name: str
    kg_co2e_per_kwh: float
    sourced: bool
    citation: str = ""
    annual_decline: float = 0.0

    def __post_init__(self) -> None:
        """Validates at construction rather than producing a null downstream.

        Follows the enforcement style of benchmarks/_result.py: a bad record
        cannot be built, so the error appears where the mistake was made.

        Raises:
          ValueError: On an empty name, a non-positive intensity, a decline
            outside [0, 1), or a sourced figure with no citation.
        """
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.kg_co2e_per_kwh, (int, float)):
            raise ValueError(
                f"kg_co2e_per_kwh must be a number, got {self.kg_co2e_per_kwh!r}"
            )
        if self.kg_co2e_per_kwh <= 0:
            raise ValueError(
                f"kg_co2e_per_kwh must be above 0, got {self.kg_co2e_per_kwh}"
            )
        if not 0.0 <= self.annual_decline < 1.0:
            raise ValueError(
                f"annual_decline must be in [0, 1), got {self.annual_decline}"
            )
        if self.sourced and not self.citation:
            # The flag exists to mark that someone did the work. Setting it
            # without a citation is exactly the silent failure it guards against.
            raise ValueError(f"{self.name} is marked sourced but carries no citation")

    def at_year(self, year: int) -> float:
        """Intensity in a given year, counting from zero.

        Args:
          year: Years from the start of the accounting window. Year 0 is now.

        Returns:
          kg CO2e per kWh in that year.

        Raises:
          ValueError: If year is negative. The model never looks backwards, so a
            negative year is a caller bug rather than a meaningful projection.
        """
        if year < 0:
            raise ValueError(f"year must be non-negative, got {year}")
        return self.kg_co2e_per_kwh * (1.0 - self.annual_decline) ** year

    def with_decline(self, annual_decline: float) -> "GridIntensity":
        """Returns a copy at a different decline rate, provenance preserved.

        Phase 9 sweeps decline. Rebuilding the object by hand each time risks
        dropping `sourced` or `citation`, so the copy is made here instead.

        Args:
          annual_decline: Fractional decline per year, in [0, 1).

        Returns:
          A new GridIntensity, same region and provenance, new decline.
        """
        return GridIntensity(
            name=self.name,
            kg_co2e_per_kwh=self.kg_co2e_per_kwh,
            sourced=self.sourced,
            citation=self.citation,
            annual_decline=annual_decline,
        )


# EPA eGRID2023 Table 1, "Subregion Output Emission Rates", CO2e total output
# column, revision 2 published June 2025. Read off the primary workbook rather
# than a summary page, and converted below. The withdrawn placeholders were
# CAISO 0.200, US average 0.390, ERCOT 0.400 and PJM 0.550; three of the four
# were high, PJM by a factor of two, so payback thresholds computed against them
# were optimistic.
EGRID_VINTAGE = "eGRID2023 rev. 2 (June 2025)"
EGRID_URL = (
    "https://www.epa.gov/system/files/documents/2025-06/summary_tables_rev2.xlsx"
)
EGRID_CITATION = (
    f"EPA {EGRID_VINTAGE}, Table 1 subregion total output emission rates, "
    f"CO2e lb/MWh. {EGRID_URL}"
)

# 1 lb = 0.45359237 kg exactly, and 1 MWh = 1000 kWh. Dropping this conversion
# is a factor of 2204.62 that still produces a plausible-looking number, which
# is the same failure shape as dropping the 3.6e6 in the carbon equation.
LB_PER_MWH_TO_KG_PER_KWH = 0.45359237 / 1000.0


def from_lb_per_mwh(lb_per_mwh: float) -> float:
    """Converts an eGRID output emission rate to the model's units.

    Args:
      lb_per_mwh: Emission rate as eGRID publishes it.

    Returns:
      The same rate in kg CO2e per kWh.

    Raises:
      ValueError: If the rate is not positive.
    """
    if lb_per_mwh <= 0:
        raise ValueError(f"lb_per_mwh must be above 0, got {lb_per_mwh}")
    return lb_per_mwh * LB_PER_MWH_TO_KG_PER_KWH


# Acronym, human name, CO2e total output lb/MWh, CO2e non-baseload lb/MWh.
# Non-baseload is eGRID's proxy for generation that responds to load changes.
_EGRID_ROWS: tuple[tuple[str, str, float, float], ...] = (
    ("CAMX", "WECC California", 429.983, 961.234),
    ("ERCT", "ERCOT All", 736.629, 1247.524),
    ("RFCE", "RFC East", 599.170, 1180.485),
    ("RFCM", "RFC Michigan", 975.978, 1517.578),
    ("RFCW", "RFC West", 916.054, 1767.963),
    ("US", "U.S. national average", 770.884, 1379.158),
)

PRESETS: dict[str, GridIntensity] = {
    acronym: GridIntensity(
        name=f"{acronym} ({label})",
        kg_co2e_per_kwh=from_lb_per_mwh(total),
        sourced=True,
        citation=EGRID_CITATION,
    )
    for acronym, label, total, _ in _EGRID_ROWS
}

# Same regions at eGRID's non-baseload rate, roughly 2x the total output rate.
# Not a default: see the module docstring on average against marginal. Phase 9
# sweeps between the two, and the gap is the size of that modelling choice.
NONBASELOAD: dict[str, GridIntensity] = {
    acronym: GridIntensity(
        name=f"{acronym} ({label}, non-baseload)",
        kg_co2e_per_kwh=from_lb_per_mwh(nonbase),
        sourced=True,
        citation=(
            f"EPA {EGRID_VINTAGE}, Table 1 subregion non-baseload output "
            f"emission rates, CO2e lb/MWh. {EGRID_URL}"
        ),
    )
    for acronym, label, _, nonbase in _EGRID_ROWS
}

# ISO and market-operator names people actually say, mapped to the eGRID
# subregion that matches. CAISO and ERCOT are near-exact; PJM is deliberately
# absent because it is not one subregion, and guessing which of the three to
# hand back would be a silent modelling decision.
_ALIASES: dict[str, str] = {
    "CAISO": "CAMX",
    "CALIFORNIA": "CAMX",
    "ERCOT": "ERCT",
    "TEXAS": "ERCT",
    "US_AVERAGE": "US",
    "US_NATIONAL": "US",
}

# PJM spans RFCE, RFCM and RFCW, which range from 0.27 to 0.44 kg CO2e/kWh.
# Naming one of them "PJM" would hide a 1.6x spread behind a label.
_PJM_SUBREGIONS = ("RFCE", "RFCM", "RFCW")


def preset(name: str) -> GridIntensity:
    """Looks up a grid region by eGRID acronym or by a common alias.

    Args:
      name: An eGRID subregion acronym such as CAMX, or an alias such as CAISO.
        Case-insensitive; spaces and hyphens are treated as underscores.

    Returns:
      The GridIntensity for that region, at the total output emission rate.

    Raises:
      KeyError: If the name is not known. Asking for PJM raises with the three
        subregions it spans, since it is not a single eGRID region.
    """
    key = name.upper().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key == "PJM":
        raise KeyError(
            "PJM is a market operator, not an eGRID subregion. It spans "
            f"{', '.join(_PJM_SUBREGIONS)}, which differ by about 1.6x, so "
            "pick one explicitly or report the range."
        )
    if key not in PRESETS:
        raise KeyError(
            f"unknown grid preset {name!r}, expected an eGRID subregion "
            f"{sorted(PRESETS)} or an alias {sorted(_ALIASES)}"
        )
    return PRESETS[key]
