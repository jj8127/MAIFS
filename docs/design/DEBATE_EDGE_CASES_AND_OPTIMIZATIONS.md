# 토론 시스템: 예외 상황 및 최적화
**Edge Cases, Exceptions, and Performance Optimizations**

---

## 🚨 발생 가능한 예외 상황

### 1. **Flip-Flopping (판정 왕복)**

#### 문제
```
Round 1: NoiseAgent AUTHENTIC → MANIPULATED (변경)
Round 2: NoiseAgent MANIPULATED → AUTHENTIC (다시 변경)
Round 3: NoiseAgent AUTHENTIC → MANIPULATED (또 변경!)
```

Agent가 계속 판정을 바꾸며 결정을 못 함

#### 원인
- LLM의 불안정한 추론
- 양측 증거가 비슷한 강도
- 프롬프트 문제

#### 해결책
```python
class FlipFlopDetector:
    """판정 왕복 탐지"""

    def __init__(self, max_flips: int = 2):
        self.max_flips = max_flips
        self.flip_history: Dict[str, List[Verdict]] = {}

    def track_verdict(self, agent_name: str, verdict: Verdict):
        if agent_name not in self.flip_history:
            self.flip_history[agent_name] = []
        self.flip_history[agent_name].append(verdict)

    def is_flip_flopping(self, agent_name: str) -> bool:
        """
        Agent가 판정을 왕복하는지 확인

        예: [AI, AUTHENTIC, AI, AUTHENTIC] → True
        """
        if agent_name not in self.flip_history:
            return False

        history = self.flip_history[agent_name]
        if len(history) < 3:
            return False

        # 최근 3개 판정에서 2번 이상 변경
        recent = history[-3:]
        changes = sum(1 for i in range(len(recent)-1) if recent[i] != recent[i+1])

        return changes >= 2

    def get_stable_verdict(self, agent_name: str) -> Verdict:
        """가장 많이 선택한 판정 반환 (다수결)"""
        from collections import Counter
        history = self.flip_history[agent_name]
        return Counter(history).most_common(1)[0][0]
```

**적용**:
```python
# debate_protocol.py

def conduct_debate(self, ...):
    flip_detector = FlipFlopDetector(max_flips=2)

    while current_round <= self.max_rounds:
        for turn in round_turns:
            flip_detector.track_verdict(turn.challenged, turn.verdict_after)

            # Flip-flopping 감지
            if flip_detector.is_flip_flopping(turn.challenged):
                # 판정 고정
                stable_verdict = flip_detector.get_stable_verdict(turn.challenged)
                responses[turn.challenged].verdict = stable_verdict

                return DebateResult(
                    termination_reason=DebateTerminationReason.FLIP_FLOPPING,
                    ...
                )
```

---

### 2. **모든 Agent가 UNCERTAIN**

#### 문제
```
Initial: [UNCERTAIN, UNCERTAIN, UNCERTAIN, UNCERTAIN]
disagreement = 0.0 (모두 같음)

→ 합의? 하지만 실제로는 "모르겠다"는 합의
```

#### 해결책
```python
def _check_consensus(self, responses: Dict[str, AgentResponse]) -> bool:
    """합의 확인 (UNCERTAIN 제외)"""
    verdicts = [r.verdict for r in responses.values()]

    # 모두 UNCERTAIN이면 합의 아님
    if all(v == Verdict.UNCERTAIN for v in verdicts):
        return False

    # UNCERTAIN 제외하고 확인
    non_uncertain = [v for v in verdicts if v != Verdict.UNCERTAIN]
    if len(set(non_uncertain)) <= 1:
        return True

    return False
```

**추가 처리**:
```python
# 모든 Agent가 UNCERTAIN일 때
if all(r.verdict == Verdict.UNCERTAIN for r in responses.values()):
    return DebateResult(
        termination_reason=DebateTerminationReason.INSUFFICIENT_EVIDENCE,
        final_verdict=Verdict.UNCERTAIN,
        confidence=0.3  # 낮은 신뢰도
    )
```

---

### 3. **Confidence Collapse (신뢰도 붕괴)**

#### 문제
```
Round 1: Agent A (confidence 0.75)
Round 2: Agent A (confidence 0.60) - 반박에 흔들림
Round 3: Agent A (confidence 0.45) - 더 흔들림
Round 4: Agent A (confidence 0.30) - 완전히 자신감 상실
```

토론 중 계속 반박당해 신뢰도가 폭락

#### 해결책
```python
class ConfidenceMonitor:
    """신뢰도 변화 추적"""

    def __init__(self, collapse_threshold: float = 0.35):
        self.collapse_threshold = collapse_threshold
        self.initial_confidence: Dict[str, float] = {}

    def track_confidence(
        self,
        agent_name: str,
        current_confidence: float,
        initial_confidence: float = None
    ):
        if initial_confidence is not None:
            self.initial_confidence[agent_name] = initial_confidence

    def is_collapsed(self, agent_name: str, current_confidence: float) -> bool:
        """신뢰도가 초기값의 50% 이하로 떨어졌는지"""
        if agent_name not in self.initial_confidence:
            return False

        initial = self.initial_confidence[agent_name]
        drop_ratio = current_confidence / initial

        # 초기 0.8 → 현재 0.3 (62% 감소)
        return drop_ratio < 0.5 or current_confidence < self.collapse_threshold

    def get_average_confidence(self, responses: Dict[str, AgentResponse]) -> float:
        """평균 신뢰도 계산"""
        confidences = [r.confidence for r in responses.values()]
        return sum(confidences) / len(confidences) if confidences else 0.0
```

**적용**:
```python
# 평균 신뢰도가 너무 낮으면 토론 중단
avg_confidence = confidence_monitor.get_average_confidence(responses)

if avg_confidence < 0.40:
    return DebateResult(
        termination_reason=DebateTerminationReason.CONFIDENCE_COLLAPSE,
        note="토론 중 신뢰도가 너무 낮아졌습니다. 증거 불충분."
    )
```

---

### 4. **LLM API 실패**

#### 문제
```
Round 2:
  Frequency → Noise: "도전..."
  Noise.respond_to_challenge() → LLM API Error!

→ 토론 중단? 재시도? 스킵?
```

#### 해결책
```python
def _execute_challenge_with_retry(
    self,
    challenger,
    challenged,
    max_retries: int = 2
) -> DebateTurn:
    """재시도 로직 포함"""

    for attempt in range(max_retries + 1):
        try:
            return self._execute_challenge(challenger, challenged, ...)

        except Exception as e:
            if attempt < max_retries:
                # 재시도
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                # 최대 재시도 초과 → Fallback
                return DebateTurn(
                    response="[LLM 오류] 판정 유지",
                    verdict_after=challenged_response.verdict,
                    confidence_after=challenged_response.confidence,
                    error=str(e)
                )
```

---

### 5. **순환 논리 (Circular Reasoning)**

#### 문제
```
Frequency: "AI_GENERATED이다. 왜냐하면 Spatial도 그렇게 말했다"
Spatial: "AI_GENERATED이다. 왜냐하면 Frequency도 그렇게 말했다"

→ 서로를 근거로 삼음 (증거 없음)
```

#### 해결책
```python
def _validate_reasoning(self, response_text: str, agent_name: str) -> bool:
    """추론의 타당성 검증"""

    # 다른 Agent 이름이 근거로 등장하는지 확인
    other_agent_names = [
        "frequency", "noise", "watermark", "spatial", "manager"
    ]
    other_agent_names.remove(agent_name.lower())

    # "~Agent가 그렇게 말했다" 같은 표현 금지
    circular_patterns = [
        f"{name} agent",
        f"{name}의 분석",
        f"{name}도"
    ]

    for pattern in circular_patterns:
        if pattern in response_text.lower():
            return False  # 순환 논리 의심

    return True
```

**프롬프트 개선**:
```python
prompt = f"""
...

⚠️ 중요: 다른 Agent의 의견을 근거로 사용하지 마세요.
오직 당신의 도구 분석 결과와 도메인 지식만을 근거로 하세요.
"""
```

---

### 6. **동점 (Tie) - 짝수 Agent**

#### 문제
```
4 Agents:
  Frequency: AI_GENERATED
  Noise: AI_GENERATED
  Watermark: AUTHENTIC
  Spatial: AUTHENTIC

→ 2 vs 2 동점!
```

#### 해결책
```python
def _break_tie(
    self,
    responses: Dict[str, AgentResponse],
    verdicts_count: Dict[Verdict, int]
) -> Verdict:
    """동점 해소"""

    # 방법 1: 신뢰도 가중 평균
    verdict_confidence = {}
    for verdict in verdicts_count.keys():
        agents_with_verdict = [
            name for name, r in responses.items() if r.verdict == verdict
        ]
        avg_conf = sum(
            responses[name].confidence for name in agents_with_verdict
        ) / len(agents_with_verdict)

        verdict_confidence[verdict] = avg_conf

    # 가장 높은 신뢰도를 가진 판정
    return max(verdict_confidence.items(), key=lambda x: x[1])[0]
```

**대안**:
```python
# 방법 2: Agent 신뢰도(trust_score) 반영
def _weighted_vote(self, responses, agent_trust):
    weighted_votes = {}
    for name, response in responses.items():
        verdict = response.verdict
        weight = agent_trust.get(name, 1.0) * response.confidence

        weighted_votes[verdict] = weighted_votes.get(verdict, 0) + weight

    return max(weighted_votes.items(), key=lambda x: x[1])[0]
```

---

## ⚡ 효율성 최적화

### 1. **선택적 토론 (Selective Debate)**

#### 문제
```
모든 이미지에 대해 토론 실행
→ LLM 비용 높음
→ 대부분은 명확한 케이스 (토론 불필요)
```

#### 해결책
```python
class SelectiveDebateStrategy:
    """토론이 필요한 케이스만 선택"""

    def should_conduct_debate(
        self,
        responses: Dict[str, AgentResponse]
    ) -> Tuple[bool, str]:
        """
        토론 필요 여부 판단

        Returns:
            (토론 필요?, 이유)
        """

        # 1. 명확한 합의 → 토론 불필요
        disagreement = self._compute_disagreement(responses)
        if disagreement < 0.2:
            return (False, "strong_consensus")

        # 2. 모두 낮은 신뢰도 → 토론해도 의미 없음
        avg_confidence = sum(r.confidence for r in responses.values()) / len(responses)
        if avg_confidence < 0.50:
            return (False, "low_confidence_all")

        # 3. 한 쪽이 압도적 → 토론 불필요
        verdicts = [r.verdict for r in responses.values()]
        from collections import Counter
        verdict_counts = Counter(verdicts)
        most_common_count = verdict_counts.most_common(1)[0][1]

        if most_common_count >= len(responses) * 0.75:  # 75% 이상
            return (False, "overwhelming_majority")

        # 4. 토론 필요
        return (True, "significant_disagreement")
```

**절감 효과**:
```
100 이미지 분석:
  - 70개: 명확한 합의 (토론 스킵)
  - 20개: 압도적 다수 (토론 스킵)
  - 10개: 실제 토론 필요

LLM 호출:
  Before: 100 × 12 = 1,200회
  After: 10 × 12 + 90 × 4 = 480회

절감: 60%
```

---

### 2. **병렬 토론 (Parallel Debate)**

#### 문제
```
Round 1:
  Frequency → Noise (순차)
  Watermark → Spatial (대기...)

→ 독립적인 쌍인데 직렬 실행
```

#### 해결책
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelDebateProtocol(DebateProtocol):
    """병렬 토론 실행"""

    async def _execute_challenge_async(self, ...):
        """비동기 도전-응답"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            turn = await loop.run_in_executor(
                executor,
                self._execute_challenge,
                challenger, challenged, ...
            )
        return turn

    async def conduct_debate_parallel(self, agents, responses):
        """병렬 토론"""
        while current_round <= self.max_rounds:
            debate_pairs = self._find_debate_pairs(responses)

            # 모든 쌍을 병렬로 실행
            tasks = [
                self._execute_challenge_async(
                    agents[a], agents[b], ...
                )
                for a, b in debate_pairs
            ]

            round_turns = await asyncio.gather(*tasks)

            # 결과 처리...
```

**성능 개선**:
```
순차 실행: 3 쌍 × 2초 = 6초
병렬 실행: max(2초, 2초, 2초) = 2초

속도: 3배 향상
```

---

### 3. **토론 히스토리 요약 (History Summarization)**

#### 문제
```
Round 5:
  Frequency → Noise: "저는..."

프롬프트에 포함:
  - Round 1~4의 모든 대화 (2,000 tokens)
  - 현재 증거 (500 tokens)

→ 총 2,500 tokens (비용 증가)
```

#### 해결책
```python
def _summarize_debate_history(
    self,
    turns: List[DebateTurn],
    max_recent: int = 2
) -> str:
    """토론 히스토리 요약"""

    if len(turns) <= max_recent:
        # 짧으면 전체 포함
        return self._format_full_history(turns)

    # 최근 N개만 상세히, 이전 것은 요약
    recent_turns = turns[-max_recent:]
    old_turns = turns[:-max_recent]

    summary = f"[이전 {len(old_turns)} 라운드 요약]\n"

    # 판정 변경만 기록
    for turn in old_turns:
        if turn.verdict_changed:
            summary += f"- Round {turn.round_number}: {turn.challenged} " \
                      f"{turn.verdict_before.value} → {turn.verdict_after.value}\n"

    summary += "\n[최근 토론]\n"
    summary += self._format_full_history(recent_turns)

    return summary
```

**토큰 절감**:
```
Before: 2,500 tokens
After: 800 tokens

비용 절감: 68%
```

---

### 4. **Confidence-Based Early Stopping**

#### 개념
```
Round 1 후:
  평균 신뢰도: 0.92
  불일치: 0.25

→ "거의 합의 + 높은 신뢰도" → 조기 종료
```

#### 구현
```python
def _check_early_stopping(
    self,
    responses: Dict[str, AgentResponse],
    disagreement: float
) -> bool:
    """조기 종료 가능 여부"""

    avg_confidence = sum(r.confidence for r in responses.values()) / len(responses)

    # 높은 신뢰도 + 낮은 불일치 → 조기 종료
    if avg_confidence >= 0.85 and disagreement <= 0.35:
        return True

    return False
```

---

### 5. **LLM 모델 선택 (Model Selection)**

#### 전략
```python
def _select_llm_model(
    self,
    disagreement: float,
    avg_confidence: float
) -> str:
    """케이스 복잡도에 따라 모델 선택"""

    # 단순 케이스: Haiku (빠르고 저렴)
    if disagreement < 0.3 and avg_confidence > 0.80:
        return "claude-haiku-4-20250514"

    # 복잡 케이스: Sonnet (균형)
    elif disagreement < 0.6:
        return "claude-sonnet-4-20250514"

    # 매우 복잡: Opus (최고 성능)
    else:
        return "claude-opus-4-5-20251101"
```

**비용 최적화**:
```
Before (모두 Sonnet):
  100 이미지 × $0.05 = $5.00

After (모델 선택):
  70 Haiku × $0.01 = $0.70
  25 Sonnet × $0.05 = $1.25
  5 Opus × $0.15 = $0.75
  Total = $2.70

절감: 46%
```

---

### 6. **캐싱 (Debate Caching)**

#### 아이디어
```
같은 증거 조합 → 같은 토론 결과

예:
  Image A: grid=0.78, prnu=0.72 → AI vs AUTHENTIC 토론
  Image B: grid=0.77, prnu=0.73 → (거의 같음) 캐시 사용
```

#### 구현
```python
import hashlib

class DebateCache:
    """토론 결과 캐싱"""

    def __init__(self, similarity_threshold: float = 0.05):
        self.cache: Dict[str, DebateResult] = {}
        self.similarity_threshold = similarity_threshold

    def _hash_evidence(self, responses: Dict[str, AgentResponse]) -> str:
        """증거 해시 생성"""
        evidence_str = ""
        for name in sorted(responses.keys()):
            r = responses[name]
            evidence_str += f"{name}:{r.verdict.value}:{r.confidence:.2f}:"
            # 주요 증거만 포함 (반올림)
            for key in sorted(r.evidence.keys()):
                val = r.evidence[key]
                if isinstance(val, float):
                    evidence_str += f"{key}:{val:.2f}:"

        return hashlib.md5(evidence_str.encode()).hexdigest()

    def get(self, responses: Dict[str, AgentResponse]) -> Optional[DebateResult]:
        """캐시 조회"""
        key = self._hash_evidence(responses)
        return self.cache.get(key)

    def put(self, responses: Dict[str, AgentResponse], result: DebateResult):
        """캐시 저장"""
        key = self._hash_evidence(responses)
        self.cache[key] = result
```

**효과**:
```
1,000 이미지 분석:
  - 200개: 유사한 패턴 (캐시 히트)

토론 횟수:
  Before: 1,000회
  After: 800회

절감: 20%
```

---

## 📊 종합 최적화 효과

```
시나리오: 1,000 이미지 분석

┌─────────────────────────────────────────────────────────┐
│ 최적화 없음                                              │
├─────────────────────────────────────────────────────────┤
│ - 모든 이미지 토론: 1,000회                              │
│ - 모두 3 라운드: 12,000 LLM 호출                         │
│ - 모델: Sonnet                                           │
│ - 비용: $600                                             │
│ - 시간: 3,000초 (50분)                                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 모든 최적화 적용                                          │
├─────────────────────────────────────────────────────────┤
│ - 선택적 토론: 300회 (70% 스킵)                          │
│ - 조기 종료: 평균 1.5 라운드                             │
│ - 캐싱: 60회 감소                                        │
│ - 모델 선택: Haiku 70%, Sonnet 25%, Opus 5%             │
│ - 병렬 실행: 2배 속도                                    │
│                                                          │
│ 결과:                                                    │
│ - LLM 호출: 1,440회 (88% 감소)                          │
│ - 비용: $95 (84% 절감)                                  │
│ - 시간: 400초 (87% 단축)                                │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ 권장 구현 순서

```
1순위: Flip-Flop 탐지 (안정성)
2순위: 선택적 토론 (비용 절감 최대)
3순위: LLM API 재시도 (안정성)
4순위: 모델 선택 (비용 절감)
5순위: 조기 종료 (효율)
6순위: 병렬 실행 (속도)
7순위: 캐싱 (효율)
```

---

이 최적화들을 적용할까요?
