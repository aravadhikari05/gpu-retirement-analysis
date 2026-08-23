# Phase 7: Embodied Carbon Estimates

Owner: Veda. Replaces the unsourced "50-400 kg CO2e" placeholder in
docs/tasks/phase8-break-even-inputs.md with sourced, method-based estimates.

## What this provides
Per-GPU embodied (manufacturing) carbon for the three swept cards, as a
LOW-HIGH range. This is `embodied_new_kg` in the Phase 8 break-even inequality.

## Headline numbers (kg CO2e per GPU)

Die-only (clean ACT method):
| GPU | die mm2 | node | low | high |
|---|---|---|---|---|
| GTX 1080 Ti | 471 | 16nm | 5.5 | 16.3 |
| RTX 2080 Ti | 754 | 12nm | 8.8 | 26.0 |
| RTX A4000 | 392 | 8nm | 4.6 | 13.6 |

Card-level (die + GDDR memory), closer to what is physically swapped:
| GPU | low | high |
|---|---|---|
| GTX 1080 Ti | 6.2 | 17.0 |
| RTX 2080 Ti | 9.5 | 26.7 |
| RTX A4000 | 5.7 | 14.6 |

## Method (ACT, Gupta et al. 2022)
Nobody publishes per-GPU embodied carbon, so we estimate from die area, the
standard approach in the sustainable-computing literature (Gupta 2022; LLMCarbon;
Toward Sustainable HPC). Manufacturing carbon scales with silicon area:

    embodied_kg = CPA * die_area_cm2 / yield  +  packaging

- CPA (carbon per area), kg CO2e/cm2: swept 1.0-3.0 as the low-high band.
  Driven by ACT's EPA (fab energy 0.8-3.5 kWh/cm2) x fab carbon intensity, plus
  process-gas and materials terms. Published lumped silicon-logic CPA clusters
  ~1-3 (Malmodin et al. ~2.6 kg/cm2). Smaller/newer nodes trend higher.
- yield = 0.875 (Gupta et al. 2022, standard).
- packaging = 150 gCO2e per IC (ACT), 1 IC per GPU.
- memory (card-level only): GDDR at 65 gCO2e/GB (ACT-derived, per LLMCarbon).

## Die sizes (sourced)
- GTX 1080 Ti: GP102, 471 mm2, TSMC 16nm. (Tom's Hardware, PCWorld.)
- RTX 2080 Ti: TU102, 754 mm2, TSMC 12nm. NVIDIA official; a physical teardown
  measured 775 mm2 (GamersNexus) - within die-measurement error, 754 used.
- RTX A4000: GA104, ~392-396 mm2, Samsung 8nm. Same die as RTX 3070.

## Scope decision needed from the team (Gap 5)
These are GPU/card figures, NOT whole-node. Vendor product carbon footprint
reports are whole-system (CPU+RAM+PSU+chassis) and run ~1000+ kg; attributing
those to one GPU is a separate modelling assumption. Because the break-even
model compares embodied against per-GPU energy savings, a GPU/card figure is
the consistent choice. Recommend: use card-level (die+GDDR), state die-only as
the floor, and note whole-node as out of scope. TEAM TO CONFIRM.

## Honesty notes for the paper
- These are estimates with wide uncertainty; the low-high band is the result,
  not the midpoint.
- The old "50-400 kg" placeholder was likely a whole-system figure; our
  die/card numbers are an order of magnitude lower BY SCOPE, not by error.
- Consequence for break-even: a smaller embodied number means replacement pays
  back SOONER than the placeholder implied - worth stating explicitly, since it
  moves the headline result.

## Sources
- Gupta et al., ACT (ISCA 2022): https://ugupta.com/files/Gupta_ISCA2022_ACT.pdf
- ACT tool: https://github.com/facebookresearch/ACT
- EPA range 0.8-3.5 kWh/cm2: Bhagavathula/Han/Gupta, HotCarbon'24
- Malmodin et al. ~2.6 kg/cm2 (via Weppe et al. 2025, 3D NAND embodied carbon)
- LLMCarbon (ICLR 2024): https://arxiv.org/pdf/2309.14393 (DRAM 65 gCO2/GB, packaging)
- Die sizes: Tom's Hardware, PCWorld, GamersNexus, NVIDIA Ampere whitepaper
