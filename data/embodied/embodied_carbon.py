"""
Phase 7: Embodied carbon estimates for the swept fleet. Owner: Veda.

Method: ACT-style (Gupta et al. 2022) area-based manufacturing carbon.
Nobody publishes per-GPU embodied carbon, so we estimate from die area x a
carbon-per-area (CPA) factor swept across the literature range, giving a
low-high band rather than a false point estimate.

  embodied_manufacturing_kg = CPA_kg_per_cm2 * die_area_cm2 / yield
  embodied_total_kg         = embodied_manufacturing_kg + packaging_kg

CPA range: the ACT logic model's carbon-per-area, driven mainly by EPA
(fab energy per area, 0.8-3.5 kWh/cm2 in ACT) x fab carbon intensity, plus
process-gas and materials terms. Published lumped CPA figures for silicon
logic cluster around ~1-3 kg CO2e/cm2 (e.g. Malmodin et al. ~2.6). Newer/
smaller nodes sit higher (more EUV/process energy per area). We sweep
1.0-3.0 and report the band; a central 2.0 is shown for reference only.

yield = 0.875 (Gupta et al. 2022, standard).
packaging = 0.150 kg per IC (ACT: 150 gCO2/IC), n_ic = 1 for a single GPU die.

Sources for die area / node are in the printed table and in EMBODIED.md.
All figures are ESTIMATES with wide uncertainty; use the range, not the point.
"""

YIELD = 0.875
PACKAGING_KG_PER_IC = 0.150
N_IC = 1
CPA_LOW, CPA_MID, CPA_HIGH = 1.0, 2.0, 3.0  # kg CO2e per cm^2

# gpu -> (die_area_mm2, process_node, architecture, source_note)
GPUS = {
    "NVIDIA-GeForce-GTX-1080-Ti": (471.0, "16nm TSMC", "Pascal GP102",
        "471 mm2 (PCWorld/Tom's Hardware, NVIDIA GP102)"),
    "NVIDIA-GeForce-RTX-2080-Ti": (754.0, "12nm TSMC", "Turing TU102",
        "754 mm2 (NVIDIA official; teardown 775 mm2)"),
    "NVIDIA-RTX-A4000":           (392.0, "8nm Samsung", "Ampere GA104",
        "~392-396 mm2 (GA104, same die as RTX 3070)"),
}

def embodied(die_mm2, cpa):
    die_cm2 = die_mm2 / 100.0            # mm^2 -> cm^2
    manuf = cpa * die_cm2 / YIELD
    pkg = N_IC * PACKAGING_KG_PER_IC
    return manuf + pkg

print(f"{'GPU':<28} {'die cm2':>8} {'node':<12} "
      f"{'low':>7} {'mid':>7} {'high':>7}   (kg CO2e)")
print("-" * 82)
rows = []
for name, (die_mm2, node, arch, src) in GPUS.items():
    lo = embodied(die_mm2, CPA_LOW)
    mid = embodied(die_mm2, CPA_MID)
    hi = embodied(die_mm2, CPA_HIGH)
    rows.append((name, die_mm2, node, arch, lo, mid, hi, src))
    print(f"{name:<28} {die_mm2/100:>8.2f} {node:<12} "
          f"{lo:>7.1f} {mid:>7.1f} {hi:>7.1f}")

print()
print("Notes:")
print(f"- yield={YIELD}, packaging={PACKAGING_KG_PER_IC*1000:.0f} gCO2e/IC x {N_IC} IC")
print(f"- CPA swept {CPA_LOW}-{CPA_HIGH} kg/cm2 (low/high band); mid={CPA_MID} for reference only")
print("- GPU-DIE-ONLY. Excludes board, GDDR memory, cooler, PCB, PSU, chassis.")
print("  A whole-CARD or whole-NODE figure would be substantially higher.")
print("- Estimates with wide uncertainty. Report the range.")

# write CSV for Phase 8
import csv
with open("embodied_carbon.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gpu_model","die_area_mm2","process_node","architecture",
                "embodied_kg_low","embodied_kg_mid","embodied_kg_high",
                "cpa_low_kg_cm2","cpa_high_kg_cm2","yield","packaging_kg","scope","source"])
    for name, die_mm2, node, arch, lo, mid, hi, src in rows:
        w.writerow([name, die_mm2, node, arch,
                    round(lo,1), round(mid,1), round(hi,1),
                    CPA_LOW, CPA_HIGH, YIELD, PACKAGING_KG_PER_IC,
                    "gpu-die-only", src])
print("\nwrote embodied_carbon.csv")

# ---------------------------------------------------------------------------
# CARD-LEVEL estimate: die + GDDR memory. Closer to "what actually gets swapped."
# DRAM embodied ~65 gCO2e/GB (LLMCarbon / functional-unit view, from ACT).
# ---------------------------------------------------------------------------
DRAM_KG_PER_GB = 0.065
VRAM_GB = {
    "NVIDIA-GeForce-GTX-1080-Ti": 11,
    "NVIDIA-GeForce-RTX-2080-Ti": 11,
    "NVIDIA-RTX-A4000": 16,
}
print("\n\n=== CARD-LEVEL (die + GDDR memory) ===")
print(f"{'GPU':<28} {'VRAM':>5} {'mem kg':>7} {'low':>7} {'mid':>7} {'high':>7}   (kg CO2e)")
print("-" * 78)
import csv
card_rows = []
for name, (die_mm2, node, arch, src) in GPUS.items():
    gb = VRAM_GB[name]
    mem = gb * DRAM_KG_PER_GB
    lo = embodied(die_mm2, CPA_LOW) + mem
    mid = embodied(die_mm2, CPA_MID) + mem
    hi = embodied(die_mm2, CPA_HIGH) + mem
    card_rows.append((name, gb, mem, lo, mid, hi))
    print(f"{name:<28} {gb:>4}G {mem:>7.2f} {lo:>7.1f} {mid:>7.1f} {hi:>7.1f}")
print("\n- Adds GDDR at 65 gCO2e/GB. Still excludes PCB, cooler, connectors, PSU share.")
print("- Whole-NODE (with CPU/RAM/chassis) from vendor PCF reports would be ~1000+ kg,")
print("  but attributing a node figure to one GPU is its own modelling assumption (Gap 5).")

with open("embodied_carbon_cardlevel.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["gpu_model","vram_gb","memory_kg","embodied_kg_low","embodied_kg_mid","embodied_kg_high","scope"])
    for name, gb, mem, lo, mid, hi in card_rows:
        w.writerow([name, gb, round(mem,2), round(lo,1), round(mid,1), round(hi,1), "die+gddr"])
print("wrote embodied_carbon_cardlevel.csv")
