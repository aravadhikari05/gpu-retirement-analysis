The Question
Nautilus has GPUs from 2018 running alongside GPUs from 2025. For each old GPU on the cluster, would it be better for the climate to keep running it, or retire it and replace it with a new one?

Why It's Not Obvious
Every GPU has 2 costs: 
Manufacturing carbon cost (the carbon that was used to make the gpu)
Operational carbon cost (electricity GPU burns while it runs). A new GPU does more computation per watt, so it wins here.

Replacing only makes sense if the electricity you save over time exceeds the manufacturing debt of the new card. Whether that happens depends on two things:
Utilization — A GPU that sits idle most of the time never pays back a replacement. The "savings" from a new GPU come from using less electricity per job. But those savings only accumulate when the GPU is actually running jobs.
Grid cleanliness — California's grid is relatively clean, so each kWh saved avoids less carbon than it would in a coal-heavy state. This makes replacement harder to justify — counterintuitive, but true.

Project Phases
The following ten phases describe the project approach in chronological order. Each phase includes what we are doing, how, what we produce, and the skills required.
Phase 1: Inventory the Fleet (Week 1)
What: Get a full census of every GPU model on Nautilus and how many of each exist.
How: Run kubectl get nodes with label selectors to extract GPU model labels from every node.
Output: A table of GPU models, counts, and node names.
Assigned: Aidan and Veda
Skills needed:
Skill
Level
Tutorial
Nautilus/Kubernetes (kubectl basics)
Familiar
NRP Basic K8s Tutorial


Phase 2: Build the Benchmark Container (Weeks 1–2)
What: Create a Docker image containing everything needed to run benchmarks and measure power: PyTorch, pynvml, and our benchmark scripts.
How: Write a Dockerfile based on an NVIDIA CUDA base image, install PyTorch and pynvml, copy in benchmark scripts, push to a container registry.
Output: A container image we can reference in pod specs.

Skills needed:
Skill
Level
Tutorial
Docker (Dockerfile, build, push)
Unknown
Docker Official Getting Started
PyTorch (install, basic imports)
Unknown
PyTorch: Learn the Basics


Phase 3: Write the Benchmark Workloads (Weeks 2–3)
What: Write three benchmark scripts that represent real workloads. Each runs a fixed amount of work (not a fixed amount of time) so we can compare total energy consumed to complete the same task across GPU generations.
The three workloads:
1. Image classification training — ResNet-50 on CIFAR-10, fixed number of batches. This is the standard benchmark used in the literature, making our results directly comparable. - Arav
2. LLM inference — Generate a fixed number of tokens from a pretrained model (e.g., GPT-2). Represents the fastest-growing real workload on Nautilus. - Aidan
3. Matrix multiplication — Pure FLOPS test, fixed problem size. Isolates raw compute from framework overhead. Serves as a sanity check. - Veda
Critical design rule: Fix the WORK, not the time. Every GPU does the identical task. We record how long it took and how much energy it used.

Skills needed:
Skill
Level
Tutorial
PyTorch (training loops, inference, pretrained models)
Unknown
PyTorch: Learn the Basics


Phase 4: Implement Power Measurement (Weeks 2–3)
What: Write a power monitoring script that runs as a background thread alongside each benchmark, sampling GPU power draw and logging it to CSV.
How: Use pynvml (the Python binding to NVIDIA's management library). A background thread polls nvmlDeviceGetPowerUsage() every 200ms, timestamps each reading, and writes to a CSV. When the benchmark finishes, the thread stops and we integrate power over time to get total joules.
Why not CodeCarbon? CodeCarbon is a black box that makes its own assumptions about PUE and grid intensity — the exact things our project is analyzing. Using pynvml gives us full transparency.
Why not nvidia-smi from bash? It works, but pynvml lets us keep everything in one Python process with synchronized timestamps.
Known limitation: nvidia-smi (and pynvml) can return cached readings on some architectures. Runs must be 30+ seconds minimum. We cite Yang et al. (2024) to show we are aware of this.

Skills needed:
Skill
Level
Tutorial
pynvml (GPU power monitoring)
Unknown
NVML Python API Tutorial
Python threading
Familiar
—


Phase 5: Set Up Persistent Storage (Week 2)
What: Create a PersistentVolumeClaim (PVC) so benchmark results survive pod death.
How: Write a PVC YAML spec, apply it with kubectl, then mount it into every benchmark pod at /results. Kubernetes pods are ephemeral — when a pod dies or gets preempted, its local filesystem is gone.

Skills needed:
Skill
Level
Tutorial
Nautilus/Kubernetes PVCs
Unknown
NRP Storage Tutorial


Phase 6: Run Benchmarks Across the Fleet (Weeks 3–5)
What: Run all three benchmarks on every GPU model in the cluster. Each benchmark runs 5–10 times per GPU model to get mean and standard deviation.
How: Use nodeSelector in the pod spec to pin each run to a specific GPU model. Loop over all models. Log the node name and timestamp for every run so we can correlate anomalies.
Output: CSV files on the PVC containing, for every run: GPU model, benchmark type, runtime (seconds), energy (joules), average power (watts), timestamps.
Practical concerns:
• Keep individual runs short (a few minutes max) so a preemption costs minutes, not hours.
• Run during low-usage hours when possible to minimize thermal interference from other pods.
• Sanity-check power readings on each GPU model before committing to a full sweep.
• If a GPU model only has 1–2 nodes, note the small sample size as a limitation.

Skills needed:
Skill
Level
Tutorial
Nautilus/Kubernetes nodeSelector & GPU scheduling
Unknown
NRP GPU Pods Guide
Bash scripting (automation loop)
Proficient
—


Phase 7: Gather Embodied Carbon Estimates (Weeks 5–6)
What: Collect manufacturing carbon estimates for each GPU generation from published literature.
How: Use Harvard's ACT model (Gupta et al., 2022), bottom-up from die area and process node. These are estimates, not exact numbers; nobody publishes per-chip manufacturing emissions. DONE 2026-08-23, see `data/embodied/`. Note: this line previously also named vendor product carbon footprint reports (Dell, HP, Lenovo) as a source to work backwards from. That is withdrawn. ACT yields the per-GPU figure directly, and subtracting a card out of a ~1000 kg whole-system total cannot recover a 6 to 27 kg quantity. See `CLAUDE.md` under "Vendor PCF reports are not a source here".
Critical design rule: Present ranges, not point estimates. Honest uncertainty is more defensible than false precision.

Skills needed:
Skill
Level
Tutorial
Literature review / reading papers
Familiar
—


Phase 8: Build the Carbon Model & Break-Even Analysis (Weeks 6–8)
What: For each old-to-new GPU replacement pair, calculate the break-even utilization threshold — how many hours of use per year before replacing is the greener choice.
How: Plug measured energy-per-job data and embodied carbon ranges into the break-even inequality. Vary grid carbon intensity (California actual, national average, coal-heavy states).
Output: Break-even charts showing, for each GPU pair: at X hours/year utilization and Y grid intensity, replacement pays off. Overlay Nautilus's actual California grid intensity.

Skills needed:
Skill
Level
Tutorial
Matplotlib / Plotly (data visualization)
Familiar
Matplotlib Tutorials
Pandas (data analysis)
Familiar
Pandas Getting Started


Phase 9: Sensitivity Analysis (Weeks 7–8)
What: Answer the question: if the manufacturing carbon estimates are wrong, do our conclusions still hold?
How: Sweep embodied carbon across the plausible range and show where the break-even answer flips. Also project forward using grid carbon intensity trends to show whether replacements that don't pay off today might pay off by 2030.

Skills needed:
Skill
Level
Tutorial
Matplotlib (multi-parameter plots)
Familiar
Matplotlib Tutorials


Phase 10: Writeup & Deliverable (Weeks 8–10)
What: Write the final report/paper and produce a practical recommendation for NRP.
Output: Guidance on which donated hardware is worth accepting and which old nodes are worth retiring, parameterized by utilization and grid location.

Skills needed:
Skill
Level
Tutorial
Git & GitHub (version control)
Familiar
GitHub Quickstart
Technical writing
Familiar
—



