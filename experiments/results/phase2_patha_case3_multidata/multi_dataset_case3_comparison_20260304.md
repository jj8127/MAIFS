# Multi-dataset 3-case Comparison (2026-03-04)

| Dataset | COBRA-only | DAAC-only | COBRA+DAAC | p(DAAC>COBRA, Wilcoxon) | p(Fusion>COBRA, Wilcoxon) | p(Fusion vs DAAC, Wilcoxon) |
|---|---:|---:|---:|---:|---:|---:|
| DS-A_Au_Tp_BigGANai | 0.2771 ± 0.0171 | 0.8353 ± 0.0282 | 0.8012 ± 0.0297 | 0.0020 | 0.0020 | 0.0020 |
| DS-B_Nature_Tp_BigGANai | 0.2738 ± 0.0177 | 0.8672 ± 0.0203 | 0.7882 ± 0.0435 | 0.0020 | 0.0020 | 0.0020 |
| DS-C_Au_IMD_BigGANai | 0.3027 ± 0.0198 | 0.8976 ± 0.0232 | 0.8423 ± 0.0236 | 0.0020 | 0.0020 | 0.0020 |
| DS-D_Nature_IMD_BigGANai | 0.2995 ± 0.0224 | 0.8634 ± 0.0151 | 0.8215 ± 0.0165 | 0.0020 | 0.0020 | 0.0020 |
| **Overall (40 runs)** | **0.2883 ± 0.0228** | **0.8659 ± 0.0309** | **0.8133 ± 0.0356** | **1.82e-12** | **1.82e-12** | **1.82e-12** |

- p-values are paired Wilcoxon signed-rank tests on matched split seeds.
