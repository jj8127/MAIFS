# 토론 시스템 설계
**Multi-Agent Debate Protocol for Image Forensics**

---

## 🎯 토론의 목적

현재 문제:
```
FrequencyAgent: AI_GENERATED (0.85)
NoiseAgent: AUTHENTIC (0.70)

→ 단순 투표: AI_GENERATED wins (신뢰도 높음)
→ 문제: NoiseAgent의 의견이 묵살됨
```

토론 후:
```
FrequencyAgent: "격자 패턴 발견"
NoiseAgent: "하지만 PRNU 있음 → 실제 사진에 AI 객체 추가한 것"

→ Manager: "혼합 이미지로 판정. 조작됨 (MANIPULATED)"
→ SpatialAgent에게 조작 영역 찾도록 요청
```

---

## 📋 토론 프로토콜

### Phase 1: 초기 분석
```
모든 Agent → 독립적으로 분석 → 판정 제출
```

**출력**:
```python
{
    "frequency": AgentResponse(verdict=AI_GENERATED, confidence=0.85),
    "noise": AgentResponse(verdict=AUTHENTIC, confidence=0.70),
    "watermark": AgentResponse(verdict=UNCERTAIN, confidence=0.50),
    "spatial": AgentResponse(verdict=AI_GENERATED, confidence=0.78)
}
```

### Phase 2: 불일치 탐지
```python
def detect_disagreement(responses: Dict[str, AgentResponse]) -> float:
    """불일치 수준 계산"""
    verdicts = [r.verdict for r in responses.values()]
    unique_verdicts = len(set(verdicts))

    # 엔트로피 기반 불일치 측정
    disagreement_level = unique_verdicts / len(verdicts)

    return disagreement_level
```

**판단 기준**:
- `disagreement_level > 0.5` → 토론 필요
- `disagreement_level ≤ 0.5` → 합의됨, 토론 불필요

### Phase 3: 토론 라운드

#### 라운드 구조
```
Round 1:
  FrequencyAgent → NoiseAgent: "왜 AUTHENTIC인가?"
  NoiseAgent → FrequencyAgent: "PRNU 있기 때문"

Round 2:
  FrequencyAgent → NoiseAgent: "PRNU 일관성이 낮은데?"
  NoiseAgent → FrequencyAgent: "혼합 이미지 가능성"

Round 3:
  Manager가 개입: "양측 의견 모두 타당. MANIPULATED로 판정"
```

---

## 🔧 구현 구조

### 1. DebateProtocol 클래스

```python
# src/debate/debate_protocol.py

from typing import Dict, List, Tuple
from dataclasses import dataclass
from ..agents.base_agent import AgentResponse
from ..tools.base_tool import Verdict


@dataclass
class DebateTurn:
    """토론의 한 턴"""
    round_number: int
    challenger: str  # Agent 이름
    challenged: str
    challenge: str  # 도전 내용
    response: str  # 응답 내용
    verdict_before: Verdict  # 응답 전 판정
    verdict_after: Verdict  # 응답 후 판정 (변경 가능)
    confidence_change: float  # 신뢰도 변화


@dataclass
class DebateResult:
    """토론 결과"""
    total_rounds: int
    turns: List[DebateTurn]
    final_verdicts: Dict[str, Verdict]
    consensus_reached: bool
    disagreement_level_before: float
    disagreement_level_after: float


class DebateProtocol:
    """토론 프로토콜 관리자"""

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    def should_debate(self, responses: Dict[str, AgentResponse]) -> bool:
        """토론 필요 여부 판단"""
        disagreement = self._compute_disagreement(responses)
        return disagreement > 0.5

    def _compute_disagreement(self, responses: Dict[str, AgentResponse]) -> float:
        """불일치 수준 계산"""
        verdicts = [r.verdict for r in responses.values()]
        unique_verdicts = len(set(verdicts))
        return unique_verdicts / len(verdicts) if verdicts else 0.0

    def conduct_debate(
        self,
        agents: Dict[str, 'BaseAgent'],
        responses: Dict[str, AgentResponse]
    ) -> DebateResult:
        """토론 진행"""

        turns = []
        current_round = 1

        # 토론 대상 쌍 찾기
        debate_pairs = self._find_debate_pairs(responses)

        while current_round <= self.max_rounds:
            round_turns = []

            for agent_a, agent_b in debate_pairs:
                # A가 B에게 도전
                turn = self._execute_challenge(
                    challenger=agents[agent_a],
                    challenged=agents[agent_b],
                    round_number=current_round,
                    all_responses=responses
                )
                round_turns.append(turn)

                # 판정 변경 확인
                if turn.verdict_after != turn.verdict_before:
                    # 판정이 바뀌면 responses 업데이트
                    responses[agent_b].verdict = turn.verdict_after
                    responses[agent_b].confidence += turn.confidence_change

            turns.extend(round_turns)

            # 합의 확인
            if self._check_consensus(responses):
                break

            current_round += 1

        return DebateResult(
            total_rounds=current_round - 1,
            turns=turns,
            final_verdicts={name: r.verdict for name, r in responses.items()},
            consensus_reached=self._check_consensus(responses),
            disagreement_level_before=self._compute_disagreement(responses),
            disagreement_level_after=self._compute_disagreement(responses)
        )

    def _find_debate_pairs(
        self,
        responses: Dict[str, AgentResponse]
    ) -> List[Tuple[str, str]]:
        """토론할 Agent 쌍 찾기"""
        pairs = []
        agent_names = list(responses.keys())

        for i, name_a in enumerate(agent_names):
            for name_b in agent_names[i+1:]:
                # 판정이 다르면 토론 대상
                if responses[name_a].verdict != responses[name_b].verdict:
                    # 신뢰도 높은 쪽이 도전자
                    if responses[name_a].confidence > responses[name_b].confidence:
                        pairs.append((name_a, name_b))
                    else:
                        pairs.append((name_b, name_a))

        return pairs

    def _execute_challenge(
        self,
        challenger: 'BaseAgent',
        challenged: 'BaseAgent',
        round_number: int,
        all_responses: Dict[str, AgentResponse]
    ) -> DebateTurn:
        """한 턴의 도전-응답 실행"""

        # 도전자의 주장
        challenge = challenger.generate_challenge(
            my_verdict=all_responses[challenger.name].verdict,
            my_evidence=all_responses[challenger.name].evidence,
            opponent_verdict=all_responses[challenged.name].verdict,
            opponent_evidence=all_responses[challenged.name].evidence
        )

        # 피도전자의 응답
        response_result = challenged.respond_to_challenge(
            challenge=challenge,
            challenger_name=challenger.name,
            my_current_verdict=all_responses[challenged.name].verdict,
            my_evidence=all_responses[challenged.name].evidence
        )

        return DebateTurn(
            round_number=round_number,
            challenger=challenger.name,
            challenged=challenged.name,
            challenge=challenge,
            response=response_result['response'],
            verdict_before=all_responses[challenged.name].verdict,
            verdict_after=response_result.get('verdict_after', all_responses[challenged.name].verdict),
            confidence_change=response_result.get('confidence_change', 0.0)
        )

    def _check_consensus(self, responses: Dict[str, AgentResponse]) -> bool:
        """합의 도달 여부 확인"""
        verdicts = [r.verdict for r in responses.values()]
        # 모든 판정이 같거나, 불일치가 매우 낮음
        return len(set(verdicts)) <= 1 or self._compute_disagreement(responses) < 0.3
```

---

## 💬 Agent의 토론 메서드

### BaseAgent에 추가할 메서드

```python
# src/agents/base_agent.py

class BaseAgent(ABC):
    """기본 Agent 클래스"""

    # 기존 메서드들...

    def generate_challenge(
        self,
        my_verdict: Verdict,
        my_evidence: Dict,
        opponent_verdict: Verdict,
        opponent_evidence: Dict
    ) -> str:
        """
        다른 Agent에게 도전

        "나는 X라고 판단했는데, 너는 왜 Y라고 하는가?"
        """
        if not hasattr(self, '_llm') or self._llm is None:
            # LLM 없으면 규칙 기반
            return self._generate_rule_based_challenge(
                my_verdict, opponent_verdict
            )

        # LLM 기반 도전
        prompt = f"""당신은 {self.name}입니다.

당신의 분석:
- 판정: {my_verdict.value}
- 증거: {json.dumps(my_evidence, indent=2)}

다른 전문가의 분석:
- 판정: {opponent_verdict.value}
- 증거: {json.dumps(opponent_evidence, indent=2)}

당신의 분석과 다른 전문가의 분석이 상충합니다.
전문가로서, 상대방의 판정에 논리적으로 도전하세요.

형식:
"당신은 {opponent_verdict.value}로 판단했지만, 저는 [핵심 증거]를 근거로 {my_verdict.value}라고 봅니다. [구체적 논거]"
"""

        return self._llm.generate(prompt)

    def respond_to_challenge(
        self,
        challenge: str,
        challenger_name: str,
        my_current_verdict: Verdict,
        my_evidence: Dict
    ) -> Dict:
        """
        도전에 응답

        Returns:
            {
                "response": str,  # 응답 내용
                "verdict_after": Verdict,  # 응답 후 판정 (변경 가능)
                "confidence_change": float  # 신뢰도 변화
            }
        """
        if not hasattr(self, '_llm') or self._llm is None:
            return self._generate_rule_based_response(challenge)

        # 도메인 지식 로드
        domain_knowledge = self._get_domain_knowledge_summary()

        prompt = f"""당신은 {self.name}입니다.

# 도메인 지식
{domain_knowledge}

# 당신의 현재 판정
- 판정: {my_current_verdict.value}
- 증거: {json.dumps(my_evidence, indent=2)}

# 다른 전문가의 도전
{challenger_name}: "{challenge}"

# 요청
1. 상대방의 주장을 분석하세요
2. 당신의 증거로 논리적으로 반박하세요
3. 상대방의 주장이 타당하다면, 판정을 수정할 수 있습니다
4. 판정을 바꾸지 않는다면, 명확한 근거를 제시하세요

# 출력 형식 (JSON)
{{
    "response": "당신의 반박 또는 인정",
    "verdict_changed": true/false,
    "new_verdict": "AI_GENERATED/AUTHENTIC/MANIPULATED/UNCERTAIN" (변경 시),
    "confidence_change": -0.1 ~ +0.1,
    "reasoning": "판정 변경 또는 유지 이유"
}}
"""

        llm_output = self._llm.generate(prompt)

        try:
            result = json.loads(llm_output)
            return {
                "response": result.get("response", ""),
                "verdict_after": Verdict(result["new_verdict"]) if result.get("verdict_changed") else my_current_verdict,
                "confidence_change": result.get("confidence_change", 0.0)
            }
        except (json.JSONDecodeError, KeyError):
            # JSON 파싱 실패 시 변경 없음
            return {
                "response": llm_output,
                "verdict_after": my_current_verdict,
                "confidence_change": 0.0
            }

    def _generate_rule_based_challenge(
        self,
        my_verdict: Verdict,
        opponent_verdict: Verdict
    ) -> str:
        """규칙 기반 도전 (LLM 없을 때)"""
        return f"제 분석 결과는 {my_verdict.value}입니다. {opponent_verdict.value}로 판단한 근거가 무엇입니까?"

    def _generate_rule_based_response(self, challenge: str) -> Dict:
        """규칙 기반 응답 (LLM 없을 때)"""
        return {
            "response": "제 분석 도구의 결과를 바탕으로 판단했습니다.",
            "verdict_after": None,  # 변경 없음
            "confidence_change": 0.0
        }

    def _get_domain_knowledge_summary(self) -> str:
        """도메인 지식 요약 (서브클래스에서 구현)"""
        return ""
```

---

## 🎭 토론 시나리오 예시

### Scenario 1: 합의 도달

```
[초기 상태]
Frequency: AI_GENERATED (0.85) - "격자 패턴 명확"
Noise: AUTHENTIC (0.70) - "PRNU 검출"
Spatial: AI_GENERATED (0.78) - "텍스처 불일치"

[Round 1]
Frequency → Noise:
  "격자 패턴이 0.78로 매우 높습니다. 왜 AUTHENTIC라고 판단하셨습니까?"

Noise → Frequency:
  "PRNU 패턴이 검출되었기 때문입니다. 하지만...
   PRNU 일관성이 0.65로 중간 수준이네요.
   실제 사진에 AI 객체가 추가된 혼합 이미지일 수 있겠습니다."
  → 판정 변경: AUTHENTIC → MANIPULATED
  → 신뢰도 조정: 0.70 → 0.75

[Round 2]
Spatial → Noise:
  "텍스처 불일치를 발견했고, 특정 영역이 의심됩니다.
   MANIPULATED 판정에 동의합니다."

Frequency → (모두):
  "격자 패턴은 추가된 객체에서 발견됐을 것입니다.
   MANIPULATED에 동의합니다."
  → 판정 변경: AI_GENERATED → MANIPULATED

[최종 합의]
모두 MANIPULATED → 토론 종료
Manager: "만장일치로 MANIPULATED 판정"
```

### Scenario 2: 합의 실패

```
[초기 상태]
Frequency: AI_GENERATED (0.92) - "명확한 격자 패턴"
Noise: AUTHENTIC (0.88) - "강한 PRNU"

[Round 1-3]
- 양측 모두 강한 증거
- 판정 변경 없음

[Manager 개입]
"양측 증거 모두 타당합니다.
 이는 고급 합성 기법이 사용된 MANIPULATED 이미지로 판정합니다.
 - 배경: 실제 사진 (PRNU 있음)
 - 객체: AI 생성 (격자 패턴 있음)"
```

---

## 📊 토론 효과

### Before (토론 없음)
```
결과: AI_GENERATED (평균 신뢰도 0.75)
근거: "여러 분석 결과 종합"
```

### After (토론 있음)
```
결과: MANIPULATED (신뢰도 0.82)
근거:
"초기에는 Frequency Agent가 AI_GENERATED로 판단했으나,
 Noise Agent가 PRNU 존재를 지적했습니다.
 토론 결과, 실제 사진에 AI로 생성한 객체를 합성한
 혼합 이미지로 판정했습니다.

 Spatial Agent가 (200, 150) 영역에서 조작 흔적을 발견하여
 이를 뒷받침합니다."
```

---

## 🔄 ManagerAgent 통합

```python
# src/agents/manager_agent.py

class ManagerAgent(BaseAgent):

    def __init__(self, ...):
        # 기존 코드...
        self.debate_protocol = DebateProtocol(max_rounds=3)

    def analyze(self, image, context=None):
        # 1. 개별 분석
        responses = self._collect_analyses(image, context)

        # 2. 토론 필요 여부
        if self.debate_protocol.should_debate(responses):
            # 3. 토론 진행
            debate_result = self.debate_protocol.conduct_debate(
                agents=self.agents,
                responses=responses
            )

            # 4. 토론 결과 반영
            responses = {
                name: resp for name, resp in responses.items()
            }  # 토론으로 업데이트된 responses

            # 5. 최종 판정
            final_verdict = self._make_final_decision_with_debate(
                responses, debate_result
            )
        else:
            # 토론 불필요
            final_verdict = self._make_final_decision(responses)

        return final_verdict
```

---

## ✅ 구현 순서

```
1. DebateProtocol 클래스 작성
   └─ src/debate/debate_protocol.py

2. BaseAgent에 토론 메서드 추가
   └─ generate_challenge()
   └─ respond_to_challenge()

3. 각 Specialist Agent에 LLM 통합
   └─ FrequencyAgent, NoiseAgent, WatermarkAgent, SpatialAgent
   └─ Knowledge Base 로드

4. ManagerAgent에 토론 시스템 통합
   └─ DebateProtocol 사용

5. 테스트 작성
   └─ tests/test_debate_system.py
```

---

이 설계로 구현을 진행할까요?
