# ARV Stage-2 Saved Models

이 디렉토리에는 Raspberry Pi 5 종단간 지연 측정을 위한 ARV stage-2 저장 모델이 들어간다.

구성:

- `arv_comp_nots_base.json`
- `arv_comp_nots_dsC.json`
- `arv_comp_nots_opensdi.json`
- `arv_comp_nots_aigenproxy.json`
- `manifest.json`

기준:

- strong MobileNetV2
- `comp_noTS` 조작 전문 모델
- richer veto `xgb_depth2`
- reverse cost
  - `manip -> auth = 6.0`
  - `auth -> manip = 2.0`
  - `non-casia harmful = x1.5`

`manifest.json`에는 각 모델의 tau와 export 메타데이터가 함께 저장된다.
