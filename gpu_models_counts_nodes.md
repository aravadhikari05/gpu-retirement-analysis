# Phase 1 Deliverable - GPU Models, Counts, and Node Names

Nautilus (NRP) cluster | 247 GPU nodes | 34 models | generated from `kubectl get nodes -l nvidia.com/gpu.product`


## OLD (2015-2019: Maxwell / Pascal / Volta / Turing)

### Quadro-M4000
- Architecture: Maxwell (2015) | Nodes: 1 | Total GPUs: 1
- Nodes (name, gpu_count): evldtn.evl.uic.edu (1)

### NVIDIA-TITAN-X-Pascal
- Architecture: Pascal (2016) | Nodes: 1 | Total GPUs: 1
- Nodes (name, gpu_count): k8s-dtn-01.uog.edu (1)

### NVIDIA-GeForce-GTX-1080
- Architecture: Pascal (2016) | Nodes: 1 | Total GPUs: 8
- Nodes (name, gpu_count): k8s-gpu-01.calit2.optiputer.net (8)

### NVIDIA-GeForce-GTX-1080-Ti
- Architecture: Pascal (2017) | Nodes: 13 | Total GPUs: 93
- Nodes (name, gpu_count): clu-fiona2.ucmerced.edu (8), fiona8-0.calit2.uci.edu (8), fiona8-1.calit2.uci.edu (6), fiona8.ucsc.edu (7), k8s-chase-ci-01.calit2.optiputer.net (7), k8s-chase-ci-02.calit2.optiputer.net (7), k8s-chase-ci-03.calit2.optiputer.net (7), k8s-chase-ci-04.calit2.optiputer.net (6), k8s-chase-ci-05.calit2.optiputer.net (8), k8s-gpu-03.sdsc.optiputer.net (8), k8s-gpu-1.ucsc.edu (7), k8s-gpu-2.ucsc.edu (8), ucm-fiona01.ucmerced.edu (6)

### Tesla-V100-SXM2-16GB
- Architecture: Volta (2017) | Nodes: 4 | Total GPUs: 32
- Nodes (name, gpu_count): chi-dgx-node01.csuchico.edu (8), chi-dgx-node02.csuchico.edu (8), chi-dgx-node03.csuchico.edu (8), chi-dgx-node04.csuchico.edu (8)

### NVIDIA-TITAN-Xp
- Architecture: Pascal (2017) | Nodes: 3 | Total GPUs: 16
- Nodes (name, gpu_count): dtn-gpu2.kreonet.net (4), k8s-ravi-01.calit2.optiputer.net (8), patternlab.calit2.optiputer.net (4)

### Tesla-V100-PCIE-16GB
- Architecture: Volta (2017) | Nodes: 3 | Total GPUs: 6
- Nodes (name, gpu_count): gpn-fiona-mizzou-7.rnet.missouri.edu (2), gpn-fiona-mizzou-8.rnet.missouri.edu (2), gpn-fiona-mizzou-9.rnet.missouri.edu (2)

### NVIDIA-GeForce-RTX-2080-Ti
- Architecture: Turing (2018) | Nodes: 15 | Total GPUs: 101
- Nodes (name, gpu_count): epic001.clemson.edu (7), fiona-prg1.cesnet.cz (6), k8s-chase-ci-10.calit2.optiputer.net (8), k8s-gpu-2.ucr.edu (6), k8s-gpu-3.ucr.edu (6), k8s-haosu-03.sdsc.optiputer.net (8), k8s-haosu-05.sdsc.optiputer.net (8), k8s-haosu-07.sdsc.optiputer.net (5), k8s-haosu-09.sdsc.optiputer.net (8), k8s-haosu-10.sdsc.optiputer.net (7), k8s-haosu-11.sdsc.optiputer.net (5), k8s-haosu-15.sdsc.optiputer.net (7), k8s-haosu-18.sdsc.optiputer.net (7), k8s-haosu-19.sdsc.optiputer.net (6), nrp-g1.nysernet.org (7)

### Tesla-V100-SXM2-32GB
- Architecture: Volta (2018) | Nodes: 9 | Total GPUs: 68
- Nodes (name, gpu_count): cph-dgx-node1.humboldt.edu (8), cph-dgx-node2.humboldt.edu (8), cph-dgx-node4.humboldt.edu (8), cph-dgx-node5.humboldt.edu (8), cph-dgx-node6.humboldt.edu (8), cph-dgx-node7.humboldt.edu (8), cph-dgx-node8.humboldt.edu (8), cph-dgx-node9.humboldt.edu (8), v100-cc-star-01.noc.ucsb.edu (4)

### Tesla-T4
- Architecture: Turing (2018) | Nodes: 3 | Total GPUs: 3
- Nodes (name, gpu_count): k8s-gen4-01.ampath.net (1), osg-houston-stashcache.nrp.internet2.edu (1), osg-sunnyvale-stashcache.nrp.internet2.edu (1)

### NVIDIA-TITAN-RTX
- Architecture: Turing (2018) | Nodes: 3 | Total GPUs: 6
- Nodes (name, gpu_count): k8s-haosu-24.sdsc.optiputer.net (2), rincewind.crbs.ucsd.edu (2), unseenu.crbs.ucsd.edu (2)

### Quadro-RTX-6000
- Architecture: Turing (2018) | Nodes: 2 | Total GPUs: 2
- Nodes (name, gpu_count): k8s-gen4-05.calit2.optiputer.net (1), moff.sdstate.edu (1)

### Quadro-RTX-8000
- Architecture: Turing (2018) | Nodes: 1 | Total GPUs: 1
- Nodes (name, gpu_count): k8s-u200-00.calit2.optiputer.net (1)


## MID (2020-2021: Ampere)

### NVIDIA-GeForce-RTX-3090
- Architecture: Ampere (2020) | Nodes: 49 | Total GPUs: 235
- Nodes (name, gpu_count): csusb-sci-103-11.csusb.edu (8), fiona-1.famu.edu (1), gpu-02.csusb.edu (7), hcc-chase-shor-c4705.unl.edu (8), hcc-chase-shor-c4709.unl.edu (8), hcc-chase-shor-c4715.unl.edu (4), hpc-nrp-g1.nmsu.edu (8), k8s-3090-01.clemson.edu (4), k8s-3090-02.clemson.edu (2), k8s-chase-ci-07.calit2.optiputer.net (6), knuron.calit2.optiputer.net (8), nautilus-ext-gpu01.fullerton.edu (8), nautilus01.hsrn.nyu.edu (3), nrp-01.laccd.edu (4), prp01.ifa.hawaii.edu (2), ry-gpu-01.sdsc.optiputer.net (8), ry-gpu-02.sdsc.optiputer.net (8), ry-gpu-03.sdsc.optiputer.net (8), ry-gpu-04.sdsc.optiputer.net (8), ry-gpu-05.sdsc.optiputer.net (8), ry-gpu-06.sdsc.optiputer.net (8), ry-gpu-07.sdsc.optiputer.net (8), ry-gpu-08.sdsc.optiputer.net (8), ry-gpu-09.sdsc.optiputer.net (8), ry-gpu-10.sdsc.optiputer.net (8), ry-gpu-11.sdsc.optiputer.net (8), ry-gpu-12.sdsc.optiputer.net (8), ry-gpu-13.sdsc.optiputer.net (8), ry-gpu-14.sdsc.optiputer.net (8), suncave-0 (2), suncave-1 (2), suncave-10 (2), suncave-11 (2), suncave-12 (2), suncave-13 (2), suncave-14 (2), suncave-15 (2), suncave-17 (2), suncave-2 (2), suncave-3 (2), suncave-4 (2), suncave-5 (2), suncave-6 (2), suncave-7 (2), suncave-8 (2), suncave-9 (2), suncave-head (1), uicnrp-fiona.evl.uic.edu (4), uicnrp-fiona2.evl.uic.edu (3)

### NVIDIA-A100-SXM4-80GB
- Architecture: Ampere (2020) | Nodes: 22 | Total GPUs: 119
- Nodes (name, gpu_count): gp-engine.beocat.ksu.edu (4), gp-engine.hpc.okstate.edu (4), gp-engine.usd.edu (4), gpengine-uams.areon.net (4), gpengine-uark.areon.net (4), gpn-fiona-mizzou-1.rnet.missouri.edu (4), gpn-fiona-mizzou-2.rnet.missouri.edu (4), gpn-fiona-mizzou-3.rnet.missouri.edu (4), gpn-fiona-mizzou-4.rnet.missouri.edu (4), gpn-fiona-mizzou-5.rnet.missouri.edu (4), gpn-fiona-mizzou-6.rnet.missouri.edu (4), hcc-gpengine-shor-c5303.unl.edu (4), node-1-1.sdsc.optiputer.net (8), node-1-2.sdsc.optiputer.net (8), node-1-3.sdsc.optiputer.net (8), node-1-4.sdsc.optiputer.net (8), node-2-2.sdsc.optiputer.net (8), node-2-3.sdsc.optiputer.net (8), node-2-4.sdsc.optiputer.net (8), pas.macc.net.internet2.edu (8), sphinx.sdstate.edu (3), tu.gp-engine.greatplains.net (4)

### NVIDIA-A100-80GB-PCIe
- Architecture: Ampere (2020) | Nodes: 9 | Total GPUs: 36
- Nodes (name, gpu_count): nautilus-it-gpu07.fullerton.edu (4), nautilus-it-gpu08.fullerton.edu (4), rci-nrp-gpu-02.sdsu.edu (4), rci-nrp-gpu-03.sdsu.edu (4), rci-nrp-gpu-04.sdsu.edu (4), rci-nrp-gpu-05.sdsu.edu (4), rci-nrp-gpu-06.sdsu.edu (4), rci-nrp-gpu-07.sdsu.edu (4), rci-nrp-gpu-08.sdsu.edu (4)

### NVIDIA-A100-PCIE-40GB
- Architecture: Ampere (2020) | Nodes: 8 | Total GPUs: 9
- Nodes (name, gpu_count): emporia.gp-argo.greatplains.net (1), gp-argo.usd.edu (1), hcc-gpn-argo-1.unl.edu (1), k8s-a100-01.suncorridor.org (2), oru.gp-argo.greatplains.net (1), ren-gp-argo-01.madren.org (1), sdsmt.gp-argo.greatplains.net (1), sdsu.gp-argo.greatplains.net (1)

### NVIDIA-A100-80GB-PCIe-MIG-1g.10gb
- Architecture: Ampere (2020) | Nodes: 1 | Total GPUs: 28 MIG slices
- Nodes (name, gpu_count): rci-nrp-gpu-01.sdsu.edu (28)

### NVIDIA-A10
- Architecture: Ampere (2021) | Nodes: 35 | Total GPUs: 279
- Nodes (name, gpu_count): gpu-01.nrp.mghpcc.org (8), gpu-02.nrp.mghpcc.org (8), gpu-03.nrp.mghpcc.org (8), gpu-04.nrp.mghpcc.org (8), gpu-05.nrp.mghpcc.org (8), gpu-06.nrp.mghpcc.org (8), gpu-07.nrp.mghpcc.org (8), gpu-08.nrp.mghpcc.org (8), gpu-09.nrp.mghpcc.org (8), gpu-11.nrp.mghpcc.org (8), gpu-12.nrp.mghpcc.org (8), gpu-13.nrp.mghpcc.org (8), gpu-14.nrp.mghpcc.org (8), gpu-15.nrp.mghpcc.org (8), gpu-16.nrp.mghpcc.org (8), gpu-17.nrp.mghpcc.org (8), gpu-18.nrp.mghpcc.org (8), hcc-nrp-shor-c5805.unl.edu (8), hcc-nrp-shor-c5809.unl.edu (8), hcc-nrp-shor-c5813.unl.edu (8), hcc-nrp-shor-c5817.unl.edu (8), hcc-nrp-shor-c5821.unl.edu (8), hcc-nrp-shor-c5825.unl.edu (8), hcc-nrp-shor-c5905.unl.edu (8), hcc-nrp-shor-c5909.unl.edu (8), hcc-nrp-shor-c5913.unl.edu (8), hcc-nrp-shor-c5917.unl.edu (8), hcc-nrp-shor-c5921.unl.edu (8), hcc-nrp-shor-c5925.unl.edu (8), hcc-nrp-shor-c6009.unl.edu (8), hcc-nrp-shor-c6013.unl.edu (7), hcc-nrp-shor-c6017.unl.edu (8), hcc-nrp-shor-c6021.unl.edu (8), hcc-nrp-shor-c6025.unl.edu (8), hcc-nrp-shor-c6029.unl.edu (8)

### NVIDIA-RTX-A6000
- Architecture: Ampere (2021) | Nodes: 10 | Total GPUs: 58
- Nodes (name, gpu_count): cenic-nrp1.hpc.cpp.edu (4), discover-nrp-01.sdccd.edu (8), gpu00.nrp.hpc.udel.edu (4), hcc-prp-c5036.unl.edu (8), hcc-prp-c5038.unl.edu (7), k8s-a6000-01.csus.edu (8), k8s-a6000-01.unm.edu (1), nautilus02.hsrn.nyu.edu (4), nrp-01.csumb.edu (8), nrp-a6000-01.csuchico.edu (6)

### NVIDIA-A40
- Architecture: Ampere (2021) | Nodes: 3 | Total GPUs: 12
- Nodes (name, gpu_count): bak-hpc1.csub.edu (8), k8s-usra-01.calit2.optiputer.net (2), rci-nautilus01.msu.montana.edu (2)

### NVIDIA-RTX-A5000
- Architecture: Ampere (2021) | Nodes: 3 | Total GPUs: 6
- Nodes (name, gpu_count): gpu-01.csusb.edu (4), nautilusg01.sci.cwru.edu (1), nautilusg02.sci.cwru.edu (1)

### NVIDIA-RTX-A4000
- Architecture: Ampere (2021) | Nodes: 2 | Total GPUs: 32
- Nodes (name, gpu_count): ry-gpu-15.sdsc.optiputer.net (16), ry-gpu-16.sdsc.optiputer.net (16)

### NVIDIA-A2
- Architecture: Ampere (2021) | Nodes: 1 | Total GPUs: 2
- Nodes (name, gpu_count): csn-nrp-node1.csun.edu (2)


## NEW (2022-2024: Ada / Hopper)

### NVIDIA-L40
- Architecture: Ada (2022) | Nodes: 17 | Total GPUs: 68
- Nodes (name, gpu_count): rci-tide-gpu-01.sdsu.edu (4), rci-tide-gpu-02.sdsu.edu (4), rci-tide-gpu-03.sdsu.edu (4), rci-tide-gpu-04.sdsu.edu (4), rci-tide-gpu-05.sdsu.edu (4), rci-tide-gpu-06.sdsu.edu (4), rci-tide-gpu-07.sdsu.edu (4), rci-tide-gpu-08.sdsu.edu (4), rci-tide-gpu-09.sdsu.edu (4), rci-tide-gpu-10.sdsu.edu (4), rci-tide-gpu-11.sdsu.edu (4), rci-tide-gpu-12.sdsu.edu (4), rci-tide-gpu-13.sdsu.edu (4), rci-tide-gpu-14.sdsu.edu (4), rci-tide-gpu-15.sdsu.edu (4), rci-tide-gpu-16.sdsu.edu (4), rci-tide-gpu-17.sdsu.edu (4)

### NVIDIA-H100-80GB-HBM3
- Architecture: Hopper (2022) | Nodes: 5 | Total GPUs: 24
- Nodes (name, gpu_count): exp-19-09.sdsc.optiputer.net (4), exp-19-10.sdsc.optiputer.net (4), exp-19-11.sdsc.optiputer.net (4), exp-19-12.sdsc.optiputer.net (4), hcc-nrp-sdc-dgx01.unl.edu (8)

### NVIDIA-GeForce-RTX-4090
- Architecture: Ada (2022) | Nodes: 5 | Total GPUs: 24
- Nodes (name, gpu_count): hcc-nrp-shor-c5226.unl.edu (4), hcc-nrp-shor-c5834.unl.edu (4), hcc-nrp-shor-c5934.unl.edu (4), hcc-nrp-shor-c6034.unl.edu (4), k8s-4090-01.calit2.optiputer.net (8)

### NVIDIA-L40S
- Architecture: Ada (2022) | Nodes: 4 | Total GPUs: 12
- Nodes (name, gpu_count): bak-hpc2.csub.edu (4), hcc-nrp-pki-c1703.unl.edu (4), hcc-nrp-pki-c1705.unl.edu (4), swan-interlink (?)

### NVIDIA-L4
- Architecture: Ada (2023) | Nodes: 6 | Total GPUs: 96
- Nodes (name, gpu_count): nautilus-it-gpu01.fullerton.edu (16), nautilus-it-gpu02.fullerton.edu (16), nautilus-it-gpu03.fullerton.edu (16), nautilus-it-gpu04.fullerton.edu (16), nautilus-it-gpu05.fullerton.edu (16), nautilus-it-gpu06.fullerton.edu (16)

### NVIDIA-H200-NVL
- Architecture: Hopper (2023) | Nodes: 3 | Total GPUs: 10
- Nodes (name, gpu_count): hcc-nrp-sec-c1109.unl.edu (4), hd200-01.ts.fresnostate.edu (2), k8s-h200-01.csus.edu (4)

### NVIDIA-GH200-480GB
- Architecture: Hopper+Grace (2023) | Nodes: 1 | Total GPUs: 1
- Nodes (name, gpu_count): gpn-fiona-mizzou-gh1.rnet.missouri.edu (1)

### NVIDIA-RTX-5000-Ada-Generation
- Architecture: Ada (2023) | Nodes: 1 | Total GPUs: 8
- Nodes (name, gpu_count): k8s-gpu-6.ucsc.edu (8)

### NVIDIA-RTX-4000-Ada-Generation
- Architecture: Ada (2023) | Nodes: 1 | Total GPUs: 2
- Nodes (name, gpu_count): stratus1.nrp-espm.berkeley.edu (2)


## NEWEST (2025: Blackwell)

### NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition
- Architecture: Blackwell (2025) | Nodes: 2 | Total GPUs: 8
- Nodes (name, gpu_count): bw6600-01.ts.fresnostate.edu (4), bw6600-02.ts.fresnostate.edu (4)
