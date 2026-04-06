# ARV End-to-End All Backends Experiment Summary

## Experiment Overview

* **Bundle:** `ARV_EndToEnd_RPi5`
* **Bundle path:** `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5`
* **Input image:** `Raspberry_pi5_Experiment/ARV_EndToEnd_RPi5/assets/benchmark_input.png`
* **Protocol:** `paper_v2` (`warmup=0`, `measured_runs=10`)
* **Device model:** `Raspberry Pi 5 Model B Rev 1.0`

---

## Notes

* `CPU`와 `USB Coral` 결과는 기존 측정값을 그대로 합산했고, `PCIe HAT` 결과는 HAT 연결 완료 후 새로 측정했다.
* HAT 재측정 시점에는 USB TPU가 연결되어 있지 않았으므로, 이번 통합 문서의 `USB Coral` 값은 재실행값이 아니라 이전 측정값이다.
* 이번 기본 입력 이미지는 세 백엔드 모두에서 4개 ARV stage-2 모델 전부 `ai_lock`으로 끝났다.
* 그래서 `real-path`와 `forced-stage2` 종단간 지연이 동일하게 측정됐고, stage-2 추가 지연은 모두 `0.0 ms`였다.
* HAT 실행 직전 수동 점검에서는 `lspci`에 `Global Unichip Corp. Coral Edge TPU`가 보였고 `/dev/apex_0`도 존재했다.
* 저장된 JSON의 `lspci_apex` 필드는 번들 내부 필터가 `apex` 문자열만 찾기 때문에 비어 있다.

---

## Backend Aggregate

| Backend     | Status                    | Threads / Device  | `w_spec` | Avg real-path E2E (ms) | Avg forced-stage2 E2E (ms) | Avg stage-2 total (ms) |
| ----------- | ------------------------- | ----------------- | -------: | ---------------------: | -------------------------: | ---------------------: |
| `CPU`       | `measured`                | `threads=4`       |      1.0 |                276.995 |                    276.995 |                    0.0 |
| `USB Coral` | `measured (previous run)` | `device=usb`      |      0.2 |                 66.102 |                     66.102 |                    0.0 |
| `PCIe HAT`  | `measured (new run)`      | `device=pcie-hat` |      0.2 |                 51.388 |                     51.388 |                    0.0 |

---

## Per-Model Results

| Backend     | ARV model    | MNV2 avg (ms) | SpecM avg (ms) | E2E avg (ms) | E2E std (ms) | E2E min-max (ms) | Final action              |
| ----------- | ------------ | ------------: | -------------: | -----------: | -----------: | ---------------- | ------------------------- |
| `CPU`       | `base`       |        163.60 |          92.54 |       256.14 |       66.974 | 173.3 - 404.6    | `ai_lock -> ai_generated` |
| `CPU`       | `dsC`        |        203.21 |         109.63 |       312.85 |      101.818 | 186.1 - 498.4    | `ai_lock -> ai_generated` |
| `CPU`       | `opensdi`    |        147.13 |         107.26 |       254.40 |       51.471 | 167.9 - 358.0    | `ai_lock -> ai_generated` |
| `CPU`       | `aigenproxy` |        184.85 |          99.74 |       284.59 |       62.294 | 180.9 - 392.9    | `ai_lock -> ai_generated` |
| `USB Coral` | `base`       |         30.54 |          35.88 |        66.38 |        1.649 | 64.6 - 70.2      | `ai_lock -> ai_generated` |
| `USB Coral` | `dsC`        |         29.76 |          36.33 |        66.09 |        1.739 | 64.4 - 71.0      | `ai_lock -> ai_generated` |
| `USB Coral` | `opensdi`    |         29.71 |          36.31 |        66.03 |        0.592 | 65.1 - 67.1      | `ai_lock -> ai_generated` |
| `USB Coral` | `aigenproxy` |         29.88 |          36.03 |        65.91 |        0.807 | 64.8 - 67.7      | `ai_lock -> ai_generated` |
| `PCIe HAT`  | `base`       |         21.61 |          29.33 |        50.95 |        0.982 | 49.8 - 53.4      | `ai_lock -> ai_generated` |
| `PCIe HAT`  | `dsC`        |         22.55 |          29.43 |        51.98 |        1.744 | 49.7 - 55.2      | `ai_lock -> ai_generated` |
| `PCIe HAT`  | `opensdi`    |         21.96 |          29.63 |        51.60 |        0.645 | 50.5 - 52.6      | `ai_lock -> ai_generated` |
| `PCIe HAT`  | `aigenproxy` |         21.85 |          29.17 |        51.02 |        0.506 | 50.3 - 51.9      | `ai_lock -> ai_generated` |

---

## Summary

* **Lowest latency backend:** `PCIe HAT`
* **Fastest average E2E latency:** `51.388 ms`
* **All backends produced identical stage-2 behavior**
* **Stage-2 additional latency:** `0.0 ms`
* **All models terminated with:** `ai_lock -> ai_generated`

---

## Alpha Weighting Experiments

### Overview

* Additional server-side experiments were conducted on the inverse-confidence weighting exponent `alpha` in the Stage-1 ICWMV fusion rule:
  * `w = 1 / confidence^alpha`
* `alpha = 0.0` corresponds to simple equal-weight averaging.
* Two kinds of alpha experiments were run:
  1. **Stage-1-only sweep** over `alpha = 0.0 ... 2.0` with `0.1` increments
  2. **Full end-to-end ARV rerun** for `alpha = 1.5`

### Stage-1 Sweep Result

Result source:
* `Server_Reproduction/ARV/data/experiments/results/paper_support/icwmv_weighting_sweep_20260329_135051.json`

Key points:

| Alpha | Avg macro-F1 | Avg corr | Avg net gain | Avg broken |
| -----: | -----------: | -------: | -----------: | ---------: |
| `0.0` | `0.9579` | `0.5313` | `1.75` | `16.5` |
| `1.0` | `0.9630` | `0.4921` | `6.5` | `10.0` |
| `1.4` | `0.9647` | `0.4599` | `7.75` | `7.0` |
| `1.5` | `0.9645` | `0.4338` | `7.25` | `6.25` |
| `2.0` | `0.9621` | `0.2680` | `4.75` | `4.5` |

Interpretation:

* Removing inverse-confidence weighting (`alpha = 0.0`) clearly degrades Stage-1 performance.
* The best **Stage-1-only** operating point in this fine-grained sweep was `alpha = 1.4`.
* `alpha = 1.5` remained very competitive and produced slightly fewer broken cases than `alpha = 1.4`.
* `alpha = 2.0` was too conservative: broken cases decreased further, but useful corrections also dropped enough to hurt macro-F1.

### Full End-to-End ARV Result for `alpha = 1.5`

Result sources:
* Scalar stage:
  * `Server_Reproduction/ARV/data/experiments/results/hema_icwmv_veto/hema_icwmv_veto_loo_cd_alpha1p5_20260329_144920.json`
* Meta warmstart stage:
  * `Server_Reproduction/ARV/data/experiments/results/hema_icwmv_veto/hema_icwmv_veto_meta_warmstart_alpha1p5_20260329_145021.json`
* Final ARV stage:
  * `Server_Reproduction/ARV/data/experiments/results/hema_icwmv_veto/comp_nots_richer_veto_alpha1p5_20260329_161903.json`

Comparison against the current paper baseline (`alpha = 1.0`):

| Stage | Metric | `alpha = 1.0` | `alpha = 1.5` |
| --- | --- | ---: | ---: |
| Scalar veto | Avg macro-F1 | `0.9622` | `0.9634` |
| Scalar veto | Avg corr | `0.3996` | `0.3771` |
| Scalar veto | Avg net gain | `6.0` | `5.75` |
| Meta warmstart | Avg macro-F1 | `0.9625` | `0.9645` |
| Meta warmstart | Avg corr | `0.4044` | `0.4199` |
| Meta warmstart | Avg net gain | `6.0` | `7.25` |
| Final ARV richer veto | Avg macro-F1 | `0.9652` | `0.9637` |
| Final ARV richer veto | Avg corr | `0.3735` | `0.3277` |
| Final ARV richer veto | Avg net gain | `8.5` | `6.25` |

Dataset-level final ARV macro-F1 comparison:

| Dataset | `alpha = 1.0` | `alpha = 1.5` |
| --- | ---: | ---: |
| `base` | `0.9586` | `0.9526` |
| `dsC` | `0.9811` | `0.9833` |
| `opensdi` | `0.9545` | `0.9500` |
| `aigenproxy` | `0.9666` | `0.9688` |

Interpretation:

* A stronger inverse-confidence weight helped the **Stage-1** fusion and also improved the **meta warmstart** intermediate result.
* However, the **final ARV richer veto** at `alpha = 1.5` was worse than the current `alpha = 1.0` paper baseline.
* Therefore, the present evidence supports the following conclusion:
  * `alpha = 1.4` is the best **Stage-1-only** operating point found so far.
  * `alpha = 1.0` remains the best **validated full end-to-end ARV** setting among completed runs.

### Current Status

* `alpha = 1.5` full pipeline: **completed**
* `alpha = 1.4` full pipeline: **not yet completed / no final JSON available at the time of this summary**

### Practical Takeaway

* For deployment latency discussion, the backend conclusion remains unchanged:
  * `PCIe HAT` is fastest, followed by `USB Coral`, then `CPU`.
* For algorithm selection:
  * If the question is **Stage-1 fusion only**, `alpha = 1.4` currently looks strongest.
  * If the question is **full ARV end-to-end performance**, `alpha = 1.0` is still the safest paper baseline until `alpha = 1.4` full rerun is completed.
