# 토론 종료 메커니즘 가이드
**Debate Termination Mechanisms for MAIFS**

---

## 🛑 종료 조건 5가지

### 1. ✅ CONSENSUS_REACHED (합의 도달)
**조건**: `disagreement < 0.3`

```python
# 예시
Round 1: [AI_GENERATED, AUTHENTIC, AI_GENERATED, UNCERTAIN]
         disagreement = 0.75 → 계속

Round 2: [AI_GENERATED, MANIPULATED, AI_GENERATED, MANIPULATED]
         disagreement = 0.50 → 계속

Round 3: [MANIPULATED, MANIPULATED, MANIPULATED, MANIPULATED]
         disagreement = 0.0 → ✅ 합의! 종료
```

**의미**: 모든 Agent가 동일하거나 거의 유사한 판정

**Manager 행동**:
```python
"전문가들이 만장일치로 MANIPULATED로 판정했습니다."
```

---

### 2. 🔄 STALEMATE (교착 상태)
**조건**: `N 라운드 동안 판정 변화 없음` (기본: N=2)

```python
Round 1:
  Frequency → Noise: "격자 패턴이 명확합니다"
  Noise → Frequency: "하지만 PRNU가 있습니다"
  → 판정 변화 없음

Round 2:
  Frequency → Noise: "격자 패턴이 명확합니다" (반복)
  Noise → Frequency: "하지만 PRNU가 있습니다" (반복)
  → 판정 변화 없음

Round 3:
  → ⚠️ STALEMATE! 2 라운드 동안 변화 없음
```

**의미**: 양측이 자신의 입장만 반복, 진전 없음

**Manager 행동**:
```python
"전문가들이 교착 상태에 도달했습니다.
 Manager가 최종 판정을 내립니다:

 Frequency Agent: AI_GENERATED (0.85) - 격자 패턴
 Noise Agent: AUTHENTIC (0.70) - PRNU 존재

 → 판정: MANIPULATED (실제 사진에 AI 객체 추가)
 → 신뢰도: 0.65 (불일치로 인한 감소)"
```

---

### 3. 🔒 HIGH_CONFIDENCE_DEADLOCK (높은 신뢰도 교착)
**조건**: 서로 다른 판정을 가진 그룹이 2개 이상, 각각 평균 신뢰도 > 0.85

```python
Round 1:
  Frequency: AI_GENERATED (0.92) - "명확한 격자 패턴"
  Noise: AUTHENTIC (0.88) - "강한 PRNU"

Round 2:
  Frequency → Noise: "격자 패턴이 0.78입니다"
  Noise: "PRNU 일관성이 0.87입니다. 판정 유지"
  → 양측 모두 확신

Round 3:
  → 🔒 HIGH_CONFIDENCE_DEADLOCK!
  → 더 토론해도 양측 모두 확신하므로 의미 없음
```

**의미**: 양측 모두 강한 증거를 가짐, 합의 불가능

**Manager 행동**:
```python
"양측 전문가 모두 높은 신뢰도로 다른 판정을 내렸습니다.
 이는 혼합 이미지(Mixed Content)의 가능성을 시사합니다.

 분석:
 - Frequency 증거: GAN 격자 패턴 (AI 생성 객체 존재)
 - Noise 증거: 카메라 PRNU (실제 촬영 배경 존재)

 → 최종 판정: MANIPULATED
 → Spatial Agent에게 조작 영역 탐지 요청"
```

---

### 4. ⏰ MAX_ROUNDS_REACHED (최대 라운드 도달)
**조건**: `current_round > max_rounds` (기본: 3)

```python
Round 1: [AI, AUTHENTIC, AI, UNCERTAIN] → 변화 있음
Round 2: [AI, MANIPULATED, AI, UNCERTAIN] → 변화 있음
Round 3: [AI, MANIPULATED, MANIPULATED, UNCERTAIN] → 변화 있음
Round 4: (X) 최대 라운드 도달

→ ⏰ MAX_ROUNDS_REACHED
```

**의미**: 진전은 있지만 시간 제한

**Manager 행동**:
```python
"최대 토론 라운드에 도달했습니다.
 현재 판정 분포:
 - AI_GENERATED: 1표
 - MANIPULATED: 2표
 - UNCERTAIN: 1표

 → 다수결: MANIPULATED
 → 신뢰도: 0.60 (불일치로 인한 감소)"
```

---

### 5. 📉 NO_PROGRESS (진전 없음)
**조건**: 판정 변화가 전혀 없음 (사실상 STALEMATE의 특수 케이스)

---

## 🎯 종료 조건 우선순위

```
1순위: CONSENSUS_REACHED (가장 이상적)
       ↓
2순위: STALEMATE (2 라운드 변화 없음)
       ↓
3순위: HIGH_CONFIDENCE_DEADLOCK (양측 확신)
       ↓
4순위: MAX_ROUNDS_REACHED (시간 제한)
```

---

## 🔧 설정 가능한 파라미터

```python
DebateProtocol(
    max_rounds=3,                    # 최대 라운드 (1~5 권장)
    consensus_threshold=0.3,         # 합의 기준 (낮을수록 엄격)
    stalemate_threshold=2,           # 교착 판정 라운드 수
    high_confidence_threshold=0.85   # 높은 신뢰도 기준
)
```

**권장 설정**:

| 시나리오 | max_rounds | consensus_threshold | stalemate_threshold |
|---------|-----------|-------------------|-------------------|
| 빠른 분석 | 2 | 0.4 | 1 |
| 기본 | 3 | 0.3 | 2 |
| 정밀 분석 | 5 | 0.2 | 3 |

---

## 📊 실제 시나리오

### Scenario 1: 빠른 합의 (이상적)
```
Initial: [AI, AI, UNCERTAIN, AI]
disagreement = 0.33

Round 1:
  Frequency → Watermark: "격자 패턴 명확"
  Watermark: "워터마크도 검출됨. AI_GENERATED로 변경"

Final: [AI, AI, AI, AI]
disagreement = 0.0

✅ CONSENSUS_REACHED (1 round)
```

### Scenario 2: 교착 상태
```
Initial: [AI (0.85), AUTHENTIC (0.80), AI (0.75), AI (0.70)]

Round 1:
  Frequency → Noise: "격자 패턴 0.78"
  Noise: "PRNU 0.82. 판정 유지"
  → 변화 없음

Round 2:
  Frequency → Noise: "여전히 격자 패턴"
  Noise: "여전히 PRNU. 판정 유지"
  → 변화 없음

🔄 STALEMATE (2 rounds without change)

Manager: "MANIPULATED (혼합 이미지)"
```

### Scenario 3: 높은 신뢰도 교착
```
Initial: [AI (0.92), AUTHENTIC (0.88)]

Round 1:
  Frequency → Noise: "격자 패턴 매우 명확"
  Noise: "PRNU 매우 강함. 판정 유지"

🔒 HIGH_CONFIDENCE_DEADLOCK

Manager: "MANIPULATED (고급 합성)"
```

### Scenario 4: 점진적 수렴 (시간 초과)
```
Initial: [AI, AUTHENTIC, UNCERTAIN, MANIPULATED]
disagreement = 1.0

Round 1: [AI, MANIPULATED, UNCERTAIN, MANIPULATED]
disagreement = 0.66 → 계속

Round 2: [AI, MANIPULATED, MANIPULATED, MANIPULATED]
disagreement = 0.33 → 계속 (아직 threshold 미만 아님)

Round 3: [MANIPULATED, MANIPULATED, MANIPULATED, MANIPULATED]
disagreement = 0.0

✅ CONSENSUS_REACHED (3 rounds)
```

---

## 💡 Manager의 종료 후 처리

```python
def handle_debate_result(self, debate_result: DebateResult, responses: Dict):
    """토론 결과 처리"""

    if debate_result.termination_reason == DebateTerminationReason.CONSENSUS_REACHED:
        # 합의 도달: 높은 신뢰도 유지
        final_verdict = list(debate_result.final_verdicts.values())[0]
        confidence = self._compute_consensus_confidence(responses)

    elif debate_result.termination_reason == DebateTerminationReason.STALEMATE:
        # 교착: Manager가 판정 + 신뢰도 감소
        final_verdict = self._manager_decision(responses, debate_result)
        confidence = 0.60  # 불일치로 인한 감소

    elif debate_result.termination_reason == DebateTerminationReason.HIGH_CONFIDENCE_DEADLOCK:
        # 높은 신뢰도 교착: MANIPULATED로 추정 + 중간 신뢰도
        final_verdict = Verdict.MANIPULATED
        confidence = 0.70
        # Spatial Agent에게 조작 영역 탐지 요청

    elif debate_result.termination_reason == DebateTerminationReason.MAX_ROUNDS_REACHED:
        # 최대 라운드: 다수결 + 낮은 신뢰도
        final_verdict = self._majority_vote(debate_result.final_verdicts)
        confidence = 0.55

    return final_verdict, confidence
```

---

## 🧪 테스트 케이스

```python
# tests/test_debate_termination.py

def test_consensus_reached():
    """합의 도달 테스트"""
    # [AI, AI, AI, AI] → CONSENSUS_REACHED

def test_stalemate():
    """교착 상태 테스트"""
    # 2 라운드 동안 변화 없음 → STALEMATE

def test_high_confidence_deadlock():
    """높은 신뢰도 교착 테스트"""
    # AI(0.92) vs AUTHENTIC(0.88) → HIGH_CONFIDENCE_DEADLOCK

def test_max_rounds():
    """최대 라운드 테스트"""
    # 3 라운드 후에도 불일치 → MAX_ROUNDS_REACHED
```

---

## ⚙️ 성능 최적화

### 1. 조기 종료로 비용 절감
```
Without early termination:
  Round 1, 2, 3 무조건 실행
  → LLM 호출 12회 (4 agents × 3 rounds)

With early termination:
  Round 1: 합의 도달 → 즉시 종료
  → LLM 호출 4회 (4 agents × 1 round)

비용 절감: 66%
```

### 2. 교착 상태 조기 감지
```
Without stalemate detection:
  같은 주장 반복 3 라운드
  → LLM 호출 12회, 의미 없는 토론

With stalemate detection:
  2 라운드 후 감지 → Manager 개입
  → LLM 호출 8회, 빠른 결론
```

---

**핵심**: 토론은 **생산적**이어야 하며, **무한 루프를 방지**하고, **비용 효율적**이어야 합니다.
