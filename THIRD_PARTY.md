# Sources and attribution

This repository packages an experimental integration built on:

- [NVIDIA cuRobo](https://github.com/NVlabs/curobo), v0.7.7, commit `0a50de1ba72db304195d59d9d0b1ed269696047f`: MPPI, robot kinematics, collision queries and PrimitiveCollisionCost. Isaac Sim / RealSense examples provided the starting structure. The modified `experiment/helper.py` retains NVIDIA notices. The complete upstream NVIDIA License is included in [LICENSE](LICENSE); the original work and derived code retain its research/evaluation, non-commercial restrictions.
- [nvblox_torch](https://github.com/NVlabs/nvblox_torch), commit `e121121c100669848c75e1be97df94a91307e23f`, and [nvblox fork](https://github.com/valtsblukis/nvblox), commit `7bda93e993594ca69c479dde64213ac35835eb84`: separately installed dependencies; no native binaries distributed.
- NVIDIA Isaac Sim4.5 and Intel RealSense D435i SDK: separate runtime dependencies.

Project work: manual observed hand-ROI patch extraction, a separate target cost connected to the actual MPPI optimizer, assumed-mass activation-distance conditions, recorded-depth replay, measured-state/reached-goal integration, log evaluation and observation-gap diagnostics. MPPI, the core collision loss and CUDA kernels were reused rather than newly developed.

The pictures show recorded real depth registered into a virtual robot workspace. They are not recordings of an actuated physical Franka. Included real-hand data and figures are the author's project records. No real contact force, safety certification or hand-mass estimator is provided.
