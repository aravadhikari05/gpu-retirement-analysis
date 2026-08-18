"""Per-GPU preflight check. Run once on each GPU model before trusting its data.

Answers, on real hardware rather than by assumption:

  - Does this torch build actually support this card's compute capability? The
    cu121 pin exists because those wheels still carry sm_61 and newer CUDA 12.8
    builds dropped Pascal. That is the claim the GTX 1080 Ti results rest on.
  - Which NVML package answered, and does it resolve
    nvmlDeviceGetTotalEnergyConsumption? Expected to be absent before Volta, so
    a 1080 Ti should report null here and an L4 should not.
  - Does NVML report plausible wattage, and at what granularity? Older consumer
    cards may quantise power in coarse steps, which has to be logged per model
    and reported rather than discovered during analysis.

Writes one JSON record to stdout prefixed with PREFLIGHT_JSON so it can be
recovered from pod logs without a PVC. Exits non-zero if a hard check fails.
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def _torch_facts() -> dict:
    """Reports what torch was built for and what it sees."""
    import torch

    facts = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "",
        "cuda_available": torch.cuda.is_available(),
        "arch_list": [],
        "device_name": "",
        "device_capability": "",
        "capability_supported": None,
    }

    try:
        import torchvision

        facts["torchvision_version"] = torchvision.__version__
    except Exception as exc:
        facts["torchvision_version"] = f"import failed: {exc}"

    try:
        import transformers

        facts["transformers_version"] = transformers.__version__
    except Exception as exc:
        facts["transformers_version"] = f"import failed: {exc}"

    if not facts["cuda_available"]:
        return facts

    facts["arch_list"] = list(torch.cuda.get_arch_list())
    facts["device_name"] = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    facts["device_capability"] = f"{major}.{minor}"

    # An exact sm_XY match is not required, and demanding one gives a false
    # negative. CUDA cubins are forward compatible across minor revisions within
    # a major generation: a cubin built for sm_60 runs on an sm_61 device, but
    # not the reverse. Measured 2026-08-18: the cu121 wheels ship sm_50, sm_60,
    # sm_70, sm_75, sm_80, sm_86, sm_90 and contain no sm_61, yet a GTX 1080 Ti
    # (sm_61) computes correctly on them via the sm_60 cubin.
    compatible = [
        arch
        for arch in facts["arch_list"]
        if arch.startswith(f"sm_{major}") and int(arch.split("_")[1][1:]) <= minor
    ]
    facts["capability_exact_match"] = f"sm_{major}{minor}" in facts["arch_list"]
    facts["capability_compatible_archs"] = compatible
    facts["capability_supported"] = bool(compatible)
    return facts


def _matmul_smoke() -> dict:
    """Confirms the card can actually compute, not just be enumerated."""
    import torch

    if not torch.cuda.is_available():
        return {"matmul_ok": False, "matmul_error": "cuda not available"}
    try:
        a = torch.rand(2048, 2048, device="cuda")
        b = torch.rand(2048, 2048, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        value = float(c.sum().item())
        del a, b, c
        torch.cuda.empty_cache()
        # A real result is finite. NaN or inf means the kernel ran on an
        # unsupported architecture and produced garbage rather than raising.
        return {
            "matmul_ok": value == value and abs(value) != float("inf"),
            "matmul_checksum": value,
        }
    except Exception as exc:
        return {"matmul_ok": False, "matmul_error": f"{type(exc).__name__}: {exc}"}


def _nvml_facts() -> dict:
    """Reports which NVML binding answered and what it supports."""
    facts = {
        "nvml_import_ok": False,
        "nvml_package": "",
        "driver_version": "",
        "nvml_device_name": "",
        "energy_counter_supported": None,
        "nvml_error": "",
    }
    try:
        import pynvml
    except Exception as exc:
        facts["nvml_error"] = f"import failed: {type(exc).__name__}: {exc}"
        return facts

    facts["nvml_import_ok"] = True
    # Both pynvml and nvidia-ml-py install a module named pynvml. Record which
    # distribution actually provided it rather than assuming.
    try:
        from importlib.metadata import packages_distributions

        providers = packages_distributions().get("pynvml", [])
        facts["nvml_package"] = ",".join(sorted(providers)) if providers else "unknown"
    except Exception:
        facts["nvml_package"] = "unknown"

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        name = pynvml.nvmlDeviceGetName(handle)
        # nvmlSystemGetDriverVersion, not nvmlDeviceGetDriverVersion. The driver
        # is a system property and the device-scoped name does not exist, so the
        # wrong one raises AttributeError and loses everything after it.
        driver = pynvml.nvmlSystemGetDriverVersion()
        facts["nvml_device_name"] = name.decode() if isinstance(name, bytes) else name
        facts["driver_version"] = (
            driver.decode() if isinstance(driver, bytes) else driver
        )

        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            facts["energy_counter_supported"] = True
        except Exception:
            # Expected before Volta. The trapezoidal integral is then the only
            # energy figure for this card, with no independent cross-check.
            facts["energy_counter_supported"] = False

        pynvml.nvmlShutdown()
    except Exception as exc:
        facts["nvml_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _power_probe(seconds: float) -> dict:
    """Samples power under load and reports plausibility and granularity."""
    from measurement.power_monitor import PowerMonitor

    facts = {"power_probe_ok": False}
    try:
        import torch

        monitor = PowerMonitor(interval=0.2)
        monitor.start()

        # Light sustained load so the reading is not idle. Not a benchmark.
        if torch.cuda.is_available():
            a = torch.rand(4096, 4096, device="cuda")
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                a = (a @ a).clamp_(-1.0, 1.0)
            torch.cuda.synchronize()
            del a
            torch.cuda.empty_cache()
        else:
            time.sleep(seconds)

        result = monitor.stop()
    except Exception as exc:
        facts["power_probe_error"] = f"{type(exc).__name__}: {exc}"
        return facts

    watts = sorted({round(r["power_w"], 3) for r in result.readings})
    # Smallest nonzero step between distinct readings. A card reporting in 25 W
    # increments shows up here as 25.0, which changes how much a small energy
    # difference between two runs can be trusted.
    steps = [round(b - a, 3) for a, b in zip(watts, watts[1:]) if b > a]

    facts.update(result.as_dict())
    facts.update(
        {
            "power_probe_ok": result.n_samples > 0,
            "distinct_power_values": len(watts),
            "min_observed_step_w": min(steps) if steps else None,
            "power_range_w": [watts[0], watts[-1]] if watts else [],
        }
    )
    return facts


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Per-GPU preflight check")
    parser.add_argument(
        "--power-seconds",
        type=float,
        default=20.0,
        help="Seconds of load while sampling power. Kept short; this is a "
        "plausibility probe, not a measured run, so the 30 s floor does not apply.",
    )
    parser.add_argument("--skip-power", action="store_true")
    args = parser.parse_args()

    record = {"hostname": platform.node(), "python_version": platform.python_version()}
    record.update(_torch_facts())
    record.update(_matmul_smoke())
    record.update(_nvml_facts())

    try:
        record["nvidia_smi"] = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,power.draw,power.limit",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:
        record["nvidia_smi"] = f"failed: {exc}"

    if not args.skip_power:
        record.update(_power_probe(args.power_seconds))

    print("PREFLIGHT_JSON " + json.dumps(record, sort_keys=True, default=str))

    # Hard failures. Anything here means data from this card is not trustworthy.
    failures = []
    if not record.get("cuda_available"):
        failures.append("cuda not available")
    if record.get("capability_supported") is False:
        failures.append(
            f"compute capability {record.get('device_capability')} is not in this "
            f"torch build's arch list {record.get('arch_list')}"
        )
    if not record.get("matmul_ok"):
        failures.append(
            f"matmul smoke failed: {record.get('matmul_error', 'bad checksum')}"
        )
    if not record.get("nvml_import_ok"):
        failures.append(f"NVML unavailable: {record.get('nvml_error')}")

    for failure in failures:
        logger.error("PREFLIGHT FAILED: %s", failure)
    if failures:
        return 1

    logger.info(
        "preflight ok: %s cap %s torch %s nvml=%s energy_counter=%s",
        record.get("device_name"),
        record.get("device_capability"),
        record.get("torch_version"),
        record.get("nvml_package"),
        record.get("energy_counter_supported"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
