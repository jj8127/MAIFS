# Multi-dataset 3-case Comparison (2026-03-04)

| Dataset | COBRA-only | DAAC-only | COBRA+DAAC | p(DAAC>COBRA, Wilcoxon) | p(Fusion>COBRA, Wilcoxon) | p(Fusion vs DAAC, Wilcoxon) |
|---|---:|---:|---:|---:|---:|---:|
| DS-A_CASIA+BigGAN | 0.2771 ± 0.0171 | 0.8353 ± 0.0282 | 0.8012 ± 0.0297 | 0.0020 | 0.0020 | 0.0020 |
| DS-B_ImageNet+CASIA+BigGAN | 0.2738 ± 0.0177 | 0.8672 ± 0.0203 | 0.7882 ± 0.0435 | 0.0020 | 0.0020 | 0.0020 |
| DS-C_CASIA+IMD2020+BigGAN | 0.3027 ± 0.0198 | 0.8976 ± 0.0232 | 0.8423 ± 0.0236 | 0.0020 | 0.0020 | 0.0020 |
| DS-D_ImageNet+IMD2020+BigGAN | 0.2995 ± 0.0224 | 0.8634 ± 0.0151 | 0.8215 ± 0.0165 | 0.0020 | 0.0020 | 0.0020 |
| OpenSDI_subset300 | 0.1772 ± 0.0121 | 0.4094 ± 0.0362 | 0.2207 ± 0.0422 | 0.0020 | 0.0137 | 0.0020 |
| AI-GenBench_proxy300 | 0.3332 ± 0.0225 | 0.7502 ± 0.0199 | 0.6168 ± 0.0248 | 0.0020 | 0.0020 | 0.0020 |
| **Overall (40 runs)** | **0.2772 ± 0.0525** | **0.7705 ± 0.1710** | **0.6818 ± 0.2229** | **1.63e-11** | **1.99e-11** | **1.63e-11** |

- p-values are paired Wilcoxon signed-rank tests on matched split seeds.
