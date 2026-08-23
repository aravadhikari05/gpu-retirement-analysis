"""Grid carbon intensity presets and their forward projection.

Separated from `carbon_model.py` because intensity is an input the model is
parameterised over, not part of the break-even arithmetic. Phase 9 sweeps it and
Phase 7 sources it; this module is where both meet the model.

**Every preset here is unsourced.** The four numbers came from a spec draft with
no citation attached, and CLAUDE.md records them as placeholders. They ship with
`sourced=False`, which propagates through the whole model and forces any output
built from them to be labelled provisional. Sourcing one is a matter of setting
`kg_co2_per_kwh`, `sourced=True` and `citation` together; the flag is not a
thing to flip on its own.

Declining intensity is modelled as a constant fractional decline per year, which
is the simplest form that captures the direction the grid is actually moving.
It matters because it works against replacement: every future year of savings
avoids less carbon than the year before, so the payback horizon stretches. A
model that held intensity constant would quietly favour replacing.
"""

from dataclasses import dataclass

# Rough annual decarbonisation rate for a US grid region, as a fraction per
# year. Unsourced like everything else here, and supplied only so the declining
# path can be exercised at all. Phase 9 owns choosing a defensible value.
EXAMPLE_ANNUAL_DECLINE = 0.03


@dataclass(frozen=True)
class GridIntensity:
    """Carbon intensity of a grid region, optionally declining over time.

    Attributes:
      name: Region label, used in output.
      kg_co2_per_kwh: Intensity in year zero.
      sourced: False until a citation is attached. Taints every figure computed
        from it, which is the point.
      citation: Where the number came from. Required when sourced is True.
      annual_decline: Fractional decline per year. 0.0 holds intensity constant,
        which makes the snapshot form a degenerate case of the integral rather
        than a separate code path.
    """

    name: str
    kg_co2_per_kwh: float
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
        if not isinstance(self.kg_co2_per_kwh, (int, float)):
            raise ValueError(
                f"kg_co2_per_kwh must be a number, got {self.kg_co2_per_kwh!r}"
            )
        if self.kg_co2_per_kwh <= 0:
            raise ValueError(
                f"kg_co2_per_kwh must be above 0, got {self.kg_co2_per_kwh}"
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
          kg CO2 per kWh in that year.

        Raises:
          ValueError: If year is negative. The model never looks backwards, so a
            negative year is a caller bug rather than a meaningful projection.
        """
        if year < 0:
            raise ValueError(f"year must be non-negative, got {year}")
        return self.kg_co2_per_kwh * (1.0 - self.annual_decline) ** year

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
            kg_co2_per_kwh=self.kg_co2_per_kwh,
            sourced=self.sourced,
            citation=self.citation,
            annual_decline=annual_decline,
        )


# All four are placeholders from a spec draft with no citation. CLAUDE.md,
# "Embodied carbon and grid intensity", records them as unsourced pending
# citation, so they are constructed that way rather than being trusted here.
PRESETS: dict[str, GridIntensity] = {
    "CAISO": GridIntensity("CAISO", 0.200, sourced=False),
    "US_AVERAGE": GridIntensity("US average", 0.390, sourced=False),
    "ERCOT": GridIntensity("ERCOT", 0.400, sourced=False),
    "PJM": GridIntensity("PJM", 0.550, sourced=False),
}


def preset(name: str) -> GridIntensity:
    """Looks up a preset by key.

    Args:
      name: One of PRESETS, case-insensitive.

    Returns:
      The GridIntensity for that region.

    Raises:
      KeyError: If the name is not a known preset, listing what is available.
    """
    key = name.upper().replace(" ", "_").replace("-", "_")
    if key not in PRESETS:
        raise KeyError(
            f"unknown grid preset {name!r}, expected one of {sorted(PRESETS)}"
        )
    return PRESETS[key]
