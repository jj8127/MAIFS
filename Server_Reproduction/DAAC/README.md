# Server Reproduction: DAAC

이 디렉토리는 DAAC의 서버 전용 재현 자산을 모아 둔 self-contained 번들이다.

## 포함 범위

- `run_daac_retrain_lightweight.py`
  - 경량 generalist 2개와 specialist 2개를 사용해 DAAC 메타 분류기를 재학습하는 스크립트
- `data/experiments/results/backbone_eval`
  - `mobilenetv2_dualstream_*`, `mobileclip_s2_finetuned_*` 최소 JSONL
- `data/experiments/results/specialist_eval`
  - `specialist_m_v2_*`, `specialist_g_*` 최소 JSONL
- `data/experiments/results/daac_retrain`
  - 기존 재학습 결과 JSON

## 빠른 시작

```bash
cd Server_Reproduction/DAAC
python3 run_daac_retrain_lightweight.py
```

## 의존성

```bash
pip install -r requirements_daac.txt
```

이 번들은 루트 `experiments/` 없이도 동작하도록 `data/experiments/results/...` 경로를 기준으로 정리되어 있다.
